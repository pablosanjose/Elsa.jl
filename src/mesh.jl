######################################################################
# Mesh
#######################################################################
abstract type AbstractMesh end
struct MeshGenerator{GV<:Base.Generator,GE<:Base.Generator,GT<:Base.Generator} <: AbstractMesh
    verts::GV
    edges::GE
    tets::GT
end
struct Mesh{D,T} <: AbstractMesh
    verts::Vector{SVector{D,T}}
    edges::Vector{Tuple{Int,Int}}
    tets::Vector{Tuple{Int,Vararg{Int,D}}}
end
Base.collect(m::MeshGenerator) = Mesh(collect(m.verts), collect(m.edges), collect(m.tets))

function marchingmesh(npoints::NTuple{D,Integer}, 
                      m::SMatrix{D,D,T} = SMatrix{D,D,Float64}(I)) where {D,T<:AbstractFloat}
    projection = m ./ (SVector(npoints) - 1)' # Projects binary vector to m box with npoints
    cs = CartesianIndices(ntuple(n -> 1:npoints[n], Val(D)))
    ls = LinearIndices(cs)
    origin = SVector(Tuple(first(cs)))
    ncs = length(cs)             # Number of verts in box (including boundary)
    csinner = CartesianIndices(ntuple(n -> 1:npoints[n]-1, Val(D)))
    ncsinner = length(csinner)   # Number of unique verts that launch edges

    # edge vectors for marching tetrahedra in D-dimensions
    uvecs = [c for c in CartesianIndices(ntuple(_ -> 0:1, Val(D)))][2:end]
    # tetrahedra built from the D unit-length uvecs added in any permutation
    perms = permutations(
            ntuple(i -> CartesianIndex(ntuple(j -> i == j ? 1 : 0, Val(D))), Val(D)))
    utets = [cumsum(pushfirst!(perm, zero(CartesianIndex{D}))) for perm in perms]

    verts = (projection * (SVector(Tuple(c)) - origin) for c in cs)
    edges = ((ls[c], ls[c + u]) for u in uvecs, c in csinner)
    tets  = (ntuple(i -> ls[c + us[i]], Val(D + 1)) for us in utets, c in csinner)
    return MeshGenerator(verts, edges, tets)
end
