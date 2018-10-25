#######################################################################
# Elements
#######################################################################

struct Elements{N,LA<:Lattice}
    lattice::LA
    indices::Matrix{Vector{SVector{N,Int}}}
end

nelements(el::Elements) = sum(length(inds) for inds in el.indices)

function Elements(lat::Lattice{T,E}) where {T,E}
    ns = nsublats(lat)
    indices = [SVector{E+1,Int}[] for _ in 1:ns, _ in 1:ns]
    return fillelements!(Elements(lat, indices))
end

Base.show(io::IO, elements::Elements{N, Lattice{T,E,L,EL}}) where {N,T,E,L,EL}=
    print(io, "Elements{$N} for $(L)D lattice in $(E)D space with $T sites
    Number of sublattices : $(nsublats(elements.lattice))
    Total number of elements : $(nelements(elements))")

function fillelements!(elem::Elements{N}) where {N} 
    isunlinked(elem.lattice) && return elem
    candidates = SVector{N,Int}[]
    buffer1 = Int[]
    buffer2 = Int[]
    for s1 in 1:size(elem.indices, 2), s2 in s1:size(elem.indices, 1)
        ind = elem.indices[s2, s1]
        _fillelements!(ind, elem.lattice.links.intralink.slinks[s2,s1], candidates, buffer1, buffer2, true)
        # for ilink in elem.lattice.links.interlinks
        #     _fillelements!(ind, ilink.slinks[s2,s1], false)
        # end
    end
    return elem
end

function _fillelements!(ind::Vector{SVector{N,Int}}, slink::Slink{T,E}, candidates, buffer1, buffer2, isintra) where {N,T,E}
    for src in sources(slink)
        resize!(candidates, 0)
        push!(candidates, modifyat(1, zero(SVector{N,Int}), src))
        imax = 0
        for pass in 2:N
            (imin, imax) = (imax + 1, length(candidates))
            # @show candidates, imin, imax
            for i in imin:imax
                neighborbuffer = common_larger_neighbors!(buffer1, buffer2, candidates[i], pass - 1, slink)
                for neigh in neighborbuffer
                    push!(candidates, modifyat(pass, candidates[i], neigh))
                end
            end
        end
        (imin, imax) = (imax + 1, length(candidates))
        for i in imin:imax
            push!(ind, candidates[i])
        end
    end
    return nothing
end

function common_larger_neighbors!(buffer1, buffer2, candidate::SVector{N,Int}, upto, slink) where {N}
    min_neighbor = maximum(candidate)
    resize!(buffer1, 0)
    resize!(buffer2, 0)
    for neigh in neighbors(slink, candidate[1])
        push!(buffer1, neigh)
    end
    for j in 2:upto
        for neigh in neighbors(slink, candidate[j])
            neigh > min_neighbor && (neigh in buffer1) && push!(buffer2, neigh)
        end
        buffer1, buffer2 = buffer2, buffer1
    end
    return buffer1
end

modifyat(i, s::SVector{N,T}, x) where {N,T} = SVector(ntuple(j -> j == i ? x : s[j], Val(N)))