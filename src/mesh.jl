######################################################################
# Mesh
#######################################################################

abstract type AbstractMesh end

struct Mesh{D,T} <: AbstractMesh
    vertices::Vector{SVector{D,T}}
    edges::Vector{Tuple{Int,Int}}
    elements::Vector{Tuple{Int,Vararg{Int,D}}}
end

# phi-space sampling z, k-space G'z. M = diagonal(partitions)
# M G' z =  Tr * n, where n are SVector{L,Int}, and Tr is a hypertriangular lattice
# Transformation D takes hypertriangularbravais to minisquare (phi-space delta): D * Tr = M^{-1}
# However, Tr has to be chosen to match bravais angles of G' as much as possible: Trsigned
# Build+link Trsigned, transform, Supercell(M)
# function unimesh(G::SMatrix{D,D,T}, partitions::NTuple{D,Int}) where {D,T}
#     M = SMatrix{D,D,T}(Diagonal(SVector(partitions)))
#     Gt = qr(transpose(G)).R
#     Gt = Gt / Gt[1,1]
#     Tr = hypertriangularbravais(lat)
#     Trsigned = zero(MMatrix{D,D,T})
#     for i in 1:D
#         Trsigned[1:L, i] .= Tr[1:L, i] * sign_positivezero(Gt[1,i])
#     end
#     D = inv(Trsigned * M)
#     meshsys = System(Bravais(Trsigned), Sublat(zero(SVector{D,T})), Model(Hopping(1))) |>
#               transform!(r -> D * r) |> grow(supercell = M)
#     vertices = meshsys.sublats[1].sites
#     edges = 
# end

# sign_positivezero(x::T) where T = x >= zero(T) ? one(T) : - one(T)

# # Completes the L2 vectors into an L-dimensional basis s with new vectors at 60 degree angles
# hypertriangularbravais(lat::Lattice{T,E,L}) where {T,E,L} = hypertriangularbravais(SMatrix{L,1,T}(I))
# function hypertriangularbravais(s::SMatrix{L,L2,T}) where {L,L2,T}
#     v1 = s[:,L2]
#     factor = T(1/(L2+1))
#     v2 = modifyat(v1, L2, v1[L2]*factor)
#     v2 = modifyat(v2, L2 + 1, v1[L2]*sqrt(1 - factor^2))
#     return hypertriangularbravais(hcat(s, v2))
# end
# hypertriangularbravais(s::SMatrix{L,L}) where L = s

# modifyat(s::SVector{N,T}, i, x) where {N,T} = SVector(ntuple(j -> j == i ? x : s[j], Val(N)))

function marchingmesh(npoints::NTuple{D,Integer}, 
                      m::SMatrix{D,D,T} = SMatrix{D,D,Float64}(I)) where {D,T<:AbstractFloat}
    projection = m ./ SVector(npoints)'
    vertices = SVector{D,T}[]
    edges = Tuple{Int,Int}[]
    elements = Tuple{Int,Vararg{Int,D}}[]
    crange = CartesianIndices(ntuple(n -> 1:npoints[n], Val(D)))
    lrange = LinearIndices(crange)
    origin = SVector(Tuple(first(crange)))

    # link vectors for marching tetrahedra in D-dimensions
    uvecs = [c for c in CartesianIndices(ntuple(_ -> 0:1, Val(D)))][2:end]

    for c in crange
        push!(vertices, projection * (SVector(Tuple(c)) - origin))
        for uvec in uvecs
            dest = c + uvec
            dest in crange && push!(edges, (lrange[c], lrange[dest]))
        end
    end
    return vertices, edges
end

# #######################################################################
# # Elements
# #######################################################################

# struct Elements{N}
#     groupsintra::Vector{Vector{SVector{N,Int}}}
#     groups::Vector{Vector{SVector{N,Int}}}
# end

# function Elements(lat::Lattice{T,E}, ::Val{N} = Val(E+1); sublat::Int = 1) where {T,E,N}
#     groupsintra = buildelementgroups(lat, sublat, Val(true), Val(N))
#     groups = buildelementgroups(lat, sublat, Val(false), Val(N))
#     return Elements(groupsintra, groups)
# end

# Base.show(io::IO, elements::Elements{N}) where {N} =
#     print(io, "Elements{$N}: groups of connected $N-vertex elements
#     Total elements     : $(nelements(elements))
#     Total groups       : $(ngroups(elements))
#     Intracell elements : $(nelementsintra(elements))
#     Intracell groups   : $(ngroupsintra(elements))")

# nelements(el::Elements) = isempty(el.groups) ? 0 : sum(length, el.groups)
# nelementsintra(el::Elements) = isempty(el.groupsintra) ? 0 : sum(length, el.groupsintra)
# ngroups(el::Elements) = length(el.groups)
# ngroupsintra(el::Elements) = length(el.groupsintra)

# function buildelementgroups(lat, sublat, onlyintra, ::Val{N}) where {N}
#     sitesubbands = siteclusters(lat, sublat, onlyintra)

#     candidatebuffer = SVector{N,Int}[]
#     buffer1 = Int[]
#     buffer2 = Int[]

#     elementgroups = [SVector{N,Int}[] for _ in sitesubbands]
#     for (s, sitesubband) in enumerate(sitesubbands), src in sitesubband
#         neighiter = NeighborIterator(lat.links, src, (sublat, sublat), onlyintra)
#         addelements!(elementgroups[s], src, neighiter, candidatebuffer, buffer1, buffer2)
#     end

#     for egroup in elementgroups
#         alignnormals!(egroup, lat.sublats[sublat].sites)
#     end

#     return elementgroups
# end
# # candidatebuffer is a list of N-elements with src as a first vertex
# # a given (src, 0, 0...) multiplies to (src, n1, 0...) and (src, n2, 0...), where n1,n2 are src neighbors
# # _common_ordered_neighbors! does this for each (src, ....), adding neighs to src to buffer1, and then looking
# # among neigh to ni at each levels for common neighbors, adding them to buffer2. Interchange and continue
# function addelements!(group::Vector{SVector{N,Int}}, src::Int, neighiter, candidatebuffer, buffer1, buffer2) where {N}
#     resize!(candidatebuffer, 0)
#     candidatebuffer = SVector{N,Int}[]
#     push!(candidatebuffer, modifyat(zero(SVector{N,Int}), 1, src))
#     imax = 0
#     for pass in 2:N
#         (imin, imax) = (imax + 1, length(candidatebuffer))
#         for i in imin:imax
#             neighborbuffer =
#                 _common_ordered_neighbors!(buffer1, buffer2, candidatebuffer[i], pass - 1, neighiter)
#             for neigh in neighborbuffer
#                 push!(candidatebuffer, modifyat(candidatebuffer[i], pass, neigh))
#             end
#         end
#     end
#     (imin, imax) = (imax + 1, length(candidatebuffer))
#     for i in imin:imax
#         push!(group, candidatebuffer[i])
#     end
#     return group
# end

# function _common_ordered_neighbors!(buffer1, buffer2, candidate::SVector{N,Int}, upto, neighiter) where {N}
#     min_neighbor = maximum(candidate)
#     resize!(buffer1, 0)
#     resize!(buffer2, 0)
#     for neigh in neighbors!(neighiter, candidate[1])
#         push!(buffer1, neigh)
#     end
#     for j in 2:upto
#         for neigh in neighbors!(neighiter, candidate[j])
#             (neigh > min_neighbor) && (neigh in buffer1) && push!(buffer2, neigh)
#         end
#         buffer1, buffer2 = buffer2, buffer1
#     end
#     return buffer1
# end

# function alignnormals!(elements::Vector{SVector{N,Int}}, sites::Vector{SVector{E,T}}) where {N,E,T}
#     for (i, element) in enumerate(elements)
#         volume = elementvolume(sites[element])
#         volume < zero(T) && (elements[i] = switchlast(element))
#     end
#     return elements
# end
# switchlast(s::SVector{N}) where {N} = SVector(ntuple(i -> i < N - 1 ? s[i] : s[2N - i - 1] , Val(N)))

# elementvolume(vs::SVector{N}) where {N} = elementvolume(ntuple(i->padright(vs[i+1] - vs[1], Val(N-1)), Val(N-1)))
# elementvolume(vs::NTuple{L,SVector{L,T}}) where {L,T} = det(hcat(vs...))

# #######################################################################
# # Mesh
# #######################################################################

# struct Mesh{T,E,L,N,EL}
#     lattice::Lattice{T,E,L,EL}
#     elements::Elements{N}
# end
# Mesh(lattice::Lattice{T,E,L}, valn::Val{N} = Val(L+1); sublat::Int = 1) where {T,E,L,N} = Mesh(lattice, Elements(lattice, valn; sublat = sublat))

# nelements(m::Mesh) = nelements(m.elements)