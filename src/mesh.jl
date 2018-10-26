#######################################################################
# Elements
#######################################################################

struct Elements{N}
    indices::Vector{SVector{N,Int}}
end

Elements(lat::Lattice, sublat = 1) = Elements(elements(lat.links.intralink.slinks[sublat,sublat]))

Base.show(io::IO, elements::Elements{N}) where {N} = 
    print(io, "Elements{$N}: $(nelements(elements)) elements ($N-vertex)")

nelements(el::Elements) = length(el.indices)

function elements(slink::Slink{T,E}) where {T,E} 
    indices = SVector{E+1,Int}[]
    isempty(slink) && return indices
    candidates = SVector{E+1,Int}[]
    buffer1 = Int[]
    buffer2 = Int[]
    for src in sources(slink)
        resize!(candidates, 0)
        push!(candidates, modifyat(zero(SVector{E+1,Int}), 1, src))
        imax = 0
        for pass in 2:E+1
            (imin, imax) = (imax + 1, length(candidates))
            for i in imin:imax
                neighborbuffer = _common_ordered_neighbors!(buffer1, buffer2, candidates[i], pass - 1, slink)
                for neigh in neighborbuffer
                    push!(candidates, modifyat(candidates[i], pass, neigh))
                end
            end
        end
        (imin, imax) = (imax + 1, length(candidates))
        for i in imin:imax
            push!(indices, candidates[i])
        end
    end
    return indices
end

function _common_ordered_neighbors!(buffer1, buffer2, candidate::SVector{N,Int}, upto, slink) where {N}
    min_neighbor = maximum(candidate)
    resize!(buffer1, 0)
    resize!(buffer2, 0)
    for neigh in neighbors(slink, candidate[1])
        push!(buffer1, neigh)
    end
    for j in 2:upto
        for neigh in neighbors(slink, candidate[j])
            (neigh > min_neighbor) && (neigh in buffer1) && push!(buffer2, neigh)
        end
        buffer1, buffer2 = buffer2, buffer1
    end
    return buffer1
end

modifyat(s::SVector{N,T}, i, x) where {N,T} = SVector(ntuple(j -> j == i ? x : s[j], Val(N)))

#######################################################################
# MeshBrillouin
#######################################################################

struct MeshBrillouin{T,E}
    lattice::Lattice{T,E,0,0}
end



