#######################################################################
# Elements
#######################################################################

struct Elements{N}
    groupsintra::Vector{Vector{SVector{N,Int}}}
    groups::Vector{Vector{SVector{N,Int}}}
end

function Elements(lat::Lattice{T,E}, ::Val{N} = Val(E+1); sublat::Int = 1) where {T,E,N} 
    groupsintra = buildelementgroups(lat, sublat, Val(true), Val(N))
    groups = buildelementgroups(lat, sublat, Val(false), Val(N))
    return Elements(groupsintra, groups)
end
        
Base.show(io::IO, elements::Elements{N}) where {N} = 
    print(io, "Elements{$N}: groups of connected $N-vertex elements
    Total elements     : $(nelements(elements)) 
    Total groups       : $(ngroups(elements)) 
    Intracell elements : $(nelementsintra(elements))
    Intracell groups   : $(ngroupsintra(elements))")

nelements(el::Elements) = isempty(el.groups) ? 0 : sum(length, el.groups)
nelementsintra(el::Elements) = isempty(el.groupsintra) ? 0 : sum(length, el.groupsintra)
ngroups(el::Elements) = length(el.groups)
ngroupsintra(el::Elements) = length(el.groupsintra)

function buildelementgroups(lat, sublat, onlyintra, ::Val{N}) where {N}
    sitesubbands = siteclusters(lat, sublat, onlyintra)

    candidatebuffer = SVector{N,Int}[]
    buffer1 = Int[]
    buffer2 = Int[]

    elementgroups = [SVector{N,Int}[] for _ in sitesubbands]
    for (s, sitesubband) in enumerate(sitesubbands), src in sitesubband
        addelements!(elementgroups[s], src, lat.links, sublat, onlyintra, candidatebuffer, buffer1, buffer2)
    end
    
    for egroup in elementgroups
        alignnormals!(egroup, lat.sublats[sublat].sites)
    end
    
    return elementgroups
end
# candidatebuffer is a list of N-elements with src as a first vertex
# a given (src, 0, 0...) multiplies to (src, n1, 0...) and (src, n2, 0...), where n1,n2 are src neighbors
# _common_ordered_neighbors! does this for each (src, ....), adding neighs to src to buffer1, and then looking
# among neigh to ni at each levels for common neighbors, adding them to buffer2. Interchange and continue
function addelements!(group::Vector{SVector{N,Int}}, src::Int, links, sublat, onlyintra, candidatebuffer, buffer1, buffer2) where {N}
    resize!(candidatebuffer, 0)
    candidatebuffer = SVector{N,Int}[]
    push!(candidatebuffer, modifyat(zero(SVector{N,Int}), 1, src))
    imax = 0
    for pass in 2:N
        (imin, imax) = (imax + 1, length(candidatebuffer))
        for i in imin:imax
            neighborbuffer = 
                _common_ordered_neighbors!(buffer1, buffer2, candidatebuffer[i], pass - 1, links, sublat, onlyintra)
            for neigh in neighborbuffer
                push!(candidatebuffer, modifyat(candidatebuffer[i], pass, neigh))
            end
        end
    end
    (imin, imax) = (imax + 1, length(candidatebuffer))
    for i in imin:imax
        push!(group, candidatebuffer[i])
    end
    return group
end

function _common_ordered_neighbors!(buffer1, buffer2, candidate::SVector{N,Int}, upto, links, sublat, onlyintra) where {N}
    min_neighbor = maximum(candidate)
    resize!(buffer1, 0)
    resize!(buffer2, 0)
    for neigh in neighbors(links, candidate[1], (sublat, sublat), onlyintra)
        push!(buffer1, neigh)
    end
    for j in 2:upto
        for neigh in neighbors(links, candidate[j], (sublat, sublat), onlyintra)
            (neigh > min_neighbor) && (neigh in buffer1) && push!(buffer2, neigh)
        end
        buffer1, buffer2 = buffer2, buffer1
    end
    return buffer1
end

# function alignnormals!(elements::Vector{SVector{N,Int}}, sites::Vector{SVector{E,T}}) where {N,E,T}    
#     switch = SVector(ntuple(i -> i < N - 1 ? i : 2N - i - 1 , Val(N)))
#     for (i, element) in enumerate(elements)
#         volume = elementvolume(ntuple(j -> padright(sites[element[j+1]] - sites[element[1]], Val(N-1)), Val(N-1)))
#         volume < 0 && (elements[i] = element[switch])
#     end
#     return elements
# end

function alignnormals!(elements::Vector{SVector{N,Int}}, sites::Vector{SVector{E,T}}) where {N,E,T}    
    for (i, element) in enumerate(elements)
        volume = elementvolume(sites[element])
        volume < zero(T) && (elements[i] = switchlast(element))
    end
    return elements
end
switchlast(s::SVector{N}) where {N} = SVector(ntuple(i -> i < N - 1 ? s[i] : s[2N - i - 1] , Val(N)))

elementvolume(vs::SVector{N}) where {N} = elementvolume(ntuple(i->padright(vs[i+1] - vs[1], Val(N-1)), Val(N-1)))
elementvolume(vs::NTuple{L,SVector{L,T}}) where {L,T} = det(hcat(vs...))

#######################################################################
# Mesh
#######################################################################

struct Mesh{T,E,L,N,EL}
    lattice::Lattice{T,E,L,EL}
    elements::Elements{N}
end
Mesh(lattice::Lattice{T,E,L}, valn::Val{N} = Val(L+1); sublat::Int = 1) where {T,E,L,N} = Mesh(lattice, Elements(lattice, valn; sublat = sublat))

nelements(m::Mesh) = nelements(m.elements)

#######################################################################
# BrillouinMesh
#######################################################################
"""
    BrillouinMesh(lat::Lattice; uniform::Bool = false, partitions = 5)

Discretizes the Brillouin zone of Lattice `lat` into hypertriangular finite 
elements, using a certain number of `partitions` per Bravais axis (accepts 
an integer or a tuple of integers, one per axis). Keyword `uniform` specifies 
the type of mesh, either "uniform" (as close to equilateral as possible) or 
"simple" (cartesian partition)

# Examples
```jldoctest
julia> BrillouinMesh(Lattice(:honeycomb), uniform = true, partitions = 200)
BrillouinMesh{Float64,2} : discretization of 2-dimensional Brillouin zone
    Mesh type  : uniform
    Vertices   : 40000 
    Partitions : (200, 200)
    3-elements : 80000
```
"""
struct BrillouinMesh{T,L,N,LL}
    mesh::Mesh{T,L,L,N,LL}
    uniform::Bool
    partitions::NTuple{L,Int}
end

function BrillouinMesh(lat::Lattice{T,E,L}; uniform::Bool = false, partitions = 5) where {T,E,L}
    partitions_tuple = tontuple(Val(L), partitions)
    if uniform
        meshlat = uniform_mesh(lat, partitions_tuple)
    else
        meshlat = simple_mesh(lat, partitions_tuple)
    end
    # wrappedmesh = wrap(mesh)
    mesh = Mesh(meshlat)
    return BrillouinMesh(mesh, uniform, partitions_tuple)
end
BrillouinMesh(sys::System; kw...) = BrillouinMesh(sys.lattice; kw...)

Base.show(io::IO, m::BrillouinMesh{T,L,N}) where {T,L,N} =
    print(io, "BrillouinMesh{$T,$L} : discretization of $L-dimensional Brillouin zone
    Mesh type  : $(m.uniform ? "uniform" : "simple")
    Vertices   : $(nsites(m.mesh.lattice)) 
    Partitions : $(m.partitions)
    $N-elements : $(nelements(m))")

nelements(m::BrillouinMesh) = nelements(m.mesh)

# phi-space sampling z, k-space G'z. M = diagonal(partitions)
# M G' z =  Tr * n, where n are SVector{L,Int}, and Tr is a hypertriangular lattice
# For some integer S = (n1,n2...), (z1, z2, z3) = I (corners of BZ).
# Hence S = round.(Tr^{-1} G' M) = supercell. Bravais are z_i for n = I, so simply S^{-1}
# Links should be fixed at the Tr level, then transform so that D * Tr = S^{-1}, and do Supercell(S)
function uniform_mesh(lat::Lattice{T,E,L}, partitions_tuple::NTuple{L,Int}) where {T,E,L}
    M = diagsmatrix(partitions_tuple)
    A = qr(bravaismatrix(lat)).R
    Gt = qr(transpose(inv(A))).R
    Gt = Gt / Gt[1,1]
    Tr = hypertriangular(lat)
    S = round.(Int, inv(Tr) * Gt * M)
    iS = inv(S)
    D = iS * inv(Tr)
    meshlat = Lattice(Sublat(zero(SVector{L,T})), Bravais(Tr), LinkRule(1))
    meshlat = transform!(meshlat, r -> D * r)
    meshlat = lattice!(meshlat, Supercell(S))
    # methlat = transform!(meshlat, r -> Gt * r)  # to go back to k space
    return meshlat
end

# Transformation D takes hypertriangular to minisquare (phi-space delta): D * Tr = M^{-1}
# However, Tr has to be chosen to match bravais angles of G' as much as possible: Trsigned
# Build+link Trsigned, transform, Supercell(M)
function simple_mesh(lat::Lattice{T,E,L}, partitions_tuple::NTuple{L,Int}) where {T,E,L}
    M = diagsmatrix(partitions_tuple)
    A = qr(bravaismatrix(lat)).R
    Gt = qr(transpose(inv(A))).R
    Gt = Gt / Gt[1,1]
    Tr = hypertriangular(lat)
    Trsigned = SMatrix{L,0,T}()
    for i in 1:L
        Trsigned = hcat(Trsigned, Tr[:,i] * sign_positivezero(Gt[1,i]))
    end
    D = inv(Trsigned * M)
    meshlat = Lattice(Sublat(zero(SVector{L,T})), Bravais(Trsigned), LinkRule(1))
    meshlat = transform!(meshlat, r -> D * r)
    meshlat = lattice!(meshlat, Supercell(M))
    return meshlat
end

hypertriangular(lat::Lattice{T,E,L}) where {T,E,L} = hypertriangular(SMatrix{L,1,T}(I))
function hypertriangular(s::SMatrix{L,L2,T}) where {L,L2,T} 
    v1 = s[:,L2]
    factor = T(1/(L2+1))
    v2 = modifyat(v1, L2, v1[L2]*factor)
    v2 = modifyat(v2, L2 + 1, v1[L2]*sqrt(1 - factor^2))
    return hypertriangular(hcat(s, v2))
end
hypertriangular(s::SMatrix{L,L}) where L = s