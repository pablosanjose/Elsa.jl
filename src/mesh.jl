######################################################################
# Mesh
#######################################################################

abstract type AbstractMesh{D} end

struct Mesh{D,V,S} <: AbstractMesh{D}   # D is dimension of parameter space
    vertices::V                         # Iterable vertex container (generator, vector,...)
    adjmat::SparseMatrixCSC{Bool,Int}   # Directed graph: only dest > src
    simplices::S                        # Iterable simplex container (generator, vector,...)
end

Mesh{D}(vertices::V, adjmat, simplices::GT) where {D,V,GT} = Mesh{D,V,GT}(vertices, adjmat, simplices)

Base.show(io::IO, mesh::Mesh{D}) where {D} = print(io,
"Mesh{$D}: mesh in $D-dimensional space
  Vertices  : $(nvertices(mesh))
  Edges     : $(nedges(mesh))
  Simplices : $(nsimplices(mesh))")

nvertices(m::Mesh) = length(m.vertices)

nsimplices(m::Mesh) = length(m.simplices)

nedges(m::Mesh) = nnz(m.adjmat)

######################################################################
# Special meshes
#######################################################################
"""
  marchingmesh(npoints::NTuple{D,Integer}[, box::SMatrix{D,D})

Creates a D-dimensional marching-tetrahedra `Mesh`. The mesh is confined to the box defined 
by the columns of `box`, and contains `npoints[i]` vertices along column i.

# External links

- Marching tetrahedra (https://en.wikipedia.org/wiki/Marching_tetrahedra) in Wikipedia
"""
function marchingmesh(npoints::NTuple{D,Integer}, 
                      box::SMatrix{D,D,T} = SMatrix{D,D,Float64}(I)) where {D,T<:AbstractFloat}
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

    vgen = (projection * (SVector(Tuple(c)) - origin) for c in cs)
    sgen = (ntuple(i -> ls[c + us[i]], Val(D + 1)) for us in utets, c in csinner)

    s = SparseMatrixBuilder{Bool}(length(cs), length(cs))
    for c in cs
        for u in uedges
            dest = c + u    # only dest > src
            dest in cs && pushtocolumn!(s, ls[dest], true)
        end
        finalisecolumn!(s)
    end
    adjmat = sparse(s)

    return Mesh{D}(vgen, adjmat, sgen)
end
