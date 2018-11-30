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
        neighiter = NeighborIterator(lat.links, src, (sublat, sublat), onlyintra)
        addelements!(elementgroups[s], src, neighiter, candidatebuffer, buffer1, buffer2)
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
function addelements!(group::Vector{SVector{N,Int}}, src::Int, neighiter, candidatebuffer, buffer1, buffer2) where {N}
    resize!(candidatebuffer, 0)
    candidatebuffer = SVector{N,Int}[]
    push!(candidatebuffer, modifyat(zero(SVector{N,Int}), 1, src))
    imax = 0
    for pass in 2:N
        (imin, imax) = (imax + 1, length(candidatebuffer))
        for i in imin:imax
            neighborbuffer =
                _common_ordered_neighbors!(buffer1, buffer2, candidatebuffer[i], pass - 1, neighiter)
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

function _common_ordered_neighbors!(buffer1, buffer2, candidate::SVector{N,Int}, upto, neighiter) where {N}
    min_neighbor = maximum(candidate)
    resize!(buffer1, 0)
    resize!(buffer2, 0)
    for neigh in neighbors!(neighiter, candidate[1])
        push!(buffer1, neigh)
    end
    for j in 2:upto
        for neigh in neighbors!(neighiter, candidate[j])
            (neigh > min_neighbor) && (neigh in buffer1) && push!(buffer2, neigh)
        end
        buffer1, buffer2 = buffer2, buffer1
    end
    return buffer1
end

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
