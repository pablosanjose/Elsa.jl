######################################################################
# Mesh
#######################################################################

abstract type AbstractMesh{D} end

struct Mesh{D,T,V<:AbstractArray{SVector{D,T}},S} <: AbstractMesh{D}   # D is dimension of parameter space
    vertices::V                         # Iterable vertex container with SVector{D,T} eltype
    adjmat::SparseMatrixCSC{Bool,Int}   # Directed graph: only dest > src
    simplices::S                        # Iterable simplex container with NTuple{D+1,Int} eltype
end

# const Mesh{D,T} = Mesh{D,T,Vector{SVector{D,T}},Vector{Tuple{Int,Vararg{Int,D}}}}

# Mesh{D,T}() where {D,T} = Mesh(SVector{D,T}[], sparse(Int[], Int[], Bool[]), NTuple{D+1,Int}[])

function Base.show(io::IO, mesh::Mesh{D}) where {D}
    i = get(io, :indent, "")
    print(io,
"$(i)Mesh{$D}: mesh of a $D-dimensional manifold
$i  Vertices   : $(nvertices(mesh))
$i  Edges      : $(nedges(mesh))
$i  Simplices  : $(nsimplices(mesh))")
end

nvertices(m::Mesh) = length(m.vertices)

nsimplices(m::Mesh) = length(m.simplices)

nedges(m::Mesh) = nnz(m.adjmat)

vertices(m::Mesh) = m.vertices

edges(m::Mesh, src) = nzrange(m.adjmat, src)

edgedest(m::Mesh, edge) = rowvals(m.adjmat)[edge]


######################################################################
# Special meshes
#######################################################################
"""
    marchingmesh(npoints::Integer...)
    marchingmesh([box::AbstractMatrix{T},] npoints::Integer....)
    marchingmesh([box::UniformScaling{T},] npoints::Integer....)

Creates a L-dimensional marching-tetrahedra `Mesh`. The mesh is confined to the box defined
by the columns of `box`, and contains `npoints[i]` vertices of type `T::AbstractFloat` along
column `i`. By default `box` is a unit box `I` along each axes. The size of `box` should
match the number `L` of elements in `npoints`.

    marchingmesh(h::Hamiltonian{<:Lattice}, npoints = 12)

Equivalent to `marchingmesh(2π*I, ntuple(_ -> npoints, Val(L))...)` where `L` is `h`'s
dimension.

# External links

- Marching tetrahedra (https://en.wikipedia.org/wiki/Marching_tetrahedra) in Wikipedia
"""
marchingmesh(npoints::Integer...) = marchingmesh(1.0 * I, npoints...)
marchingmesh(box::UniformScaling{T}, npoints::Vararg{Integer,L}) where {L,T} =
    _marchingmesh(SMatrix{L,L}(box), npoints)
marchingmesh(box::AbstractMatrix{T}, npoints::Vararg{Integer,L}) where {L,T} =
    _marchingmesh(convert(SMatrix{L,L,T}, box), npoints)
marchingmesh(h::Hamiltonian{<:Lattice,L,M}, n::Integer) where {L,M} =
    _marchingmesh(SMatrix{L,L}(one(real(eltype(M))) * 2π * I), ntuple(_ -> n, Val(L)))

function _marchingmesh(box::SMatrix{D,D,T}, npoints::NTuple{D,Integer}) where {D,T<:AbstractFloat}
    projection = box ./ (SVector(npoints) - 1)' # Projects binary vector to m box with npoints
    cs = CartesianIndices(ntuple(n -> 1:npoints[n], Val(D)))
    ls = LinearIndices(cs)
    origin = SVector(Tuple(first(cs)))
    csinner = CartesianIndices(ntuple(n -> 1:npoints[n]-1, Val(D)))

    # edge vectors for marching tetrahedra in D-dimensions
    uedges = [c for c in CartesianIndices(ntuple(_ -> 0:1, Val(D)))][2:end]
    # tetrahedra built from the D unit-length uvecs added in any permutation
    perms = permutations(
            ntuple(i -> CartesianIndex(ntuple(j -> i == j ? 1 : 0, Val(D))), Val(D)))
    utets = [cumsum(pushfirst!(perm, zero(CartesianIndex{D}))) for perm in perms]

    # We don't use generators because their non-inferreble eltype causes problems later
    verts = [projection * (SVector(Tuple(c)) - origin) for c in cs]
    simps = [ntuple(i -> ls[c + us[i]], Val(D + 1)) for us in utets, c in csinner]

    alignnormals!(simps, verts)

    s = SparseMatrixBuilder{Bool}(length(cs), length(cs))
    for c in cs
        for u in uedges
            dest = c + u    # only dest > src
            dest in cs && pushtocolumn!(s, ls[dest], true)
        end
        finalizecolumn!(s)
    end
    adjmat = sparse(s)

    return Mesh(verts, adjmat, simps)
end

function alignnormals!(simplices, vertices)
    for (i, s) in enumerate(simplices)
        volume = elementvolume(vertices, s)
        volume < 0 && (simplices[i] = switchlast(s))
    end
    return simplices
end

elementvolume(verts, s::NTuple{N,Int}) where {N} =
    elementvolume(hcat(ntuple(i -> SVector(verts[s[i+1]] - verts[s[1]]), Val(N-1))...))
elementvolume(mat::SMatrix{N,N}) where {N} = det(qr(mat).R)
elementvolume(mat::SMatrix{M,N}) where {M,N} = det(mat)

switchlast(s::NTuple{N,T}) where {N,T} = ntuple(i -> i < N - 1 ? s[i] : s[2N - i - 1] , Val(N))
