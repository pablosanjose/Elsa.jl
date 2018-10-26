#######################################################################
# Elements
#######################################################################

struct Elements{N}
    indices::Matrix{Vector{SVector{N,Int}}}
end

Elements(lat::Lattice) = Elements(elements(lat))

function Base.show(io::IO, elements::Elements{N}) where {N,T,E,L,EL}
    ns = nsublats(elements)
    print(io, "Elements{$N}: $(nelements(elements)) elements ($N-vertex) in $ns sublattice$(ns > 1 ? "s" : "")")
end

nelements(el::Elements) = sum(length(inds) for inds in el.indices)
nsublats(el::Elements) = size(el.indices, 1)

function elements(lattice::Lattice{T,E}) where {T,E}
    ns = nsublats(lattice)
    indices = [SVector{E+1,Int}[] for _ in 1:ns, _ in 1:ns]
    _elements!(indices, lattice)
    return indices
end

function _elements!(indices::Matrix{Vector{SVector{N,Int}}}, lattice) where {N} 
    isunlinked(lattice) && return nothing
    candidates = SVector{N,Int}[]
    buffer1 = Int[]
    buffer2 = Int[]
    for s1 in 1:size(indices, 2), s2 in s1:size(indices, 1)
        ind = indices[s2, s1]
        _fillelements!(ind, lattice.links.intralink.slinks[s2,s1], candidates, buffer1, buffer2, true)
    end
    return nothing
end

function _fillelements!(ind::Vector{SVector{N,Int}}, slink::Slink{T,E}, candidates, buffer1, buffer2, isintra) where {N,T,E}
    for src in sources(slink)
        resize!(candidates, 0)
        push!(candidates, modifyat(zero(SVector{N,Int}), 1, src))
        imax = 0
        for pass in 2:N
            (imin, imax) = (imax + 1, length(candidates))
            # @show candidates, imin, imax
            for i in imin:imax
                neighborbuffer = _common_larger_neighbors!(buffer1, buffer2, candidates[i], pass - 1, slink)
                for neigh in neighborbuffer
                    push!(candidates, modifyat(candidates[i], pass, neigh))
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

function _common_larger_neighbors!(buffer1, buffer2, candidate::SVector{N,Int}, upto, slink) where {N}
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

modifyat(s::SVector{N,T}, i, x) where {N,T} = SVector(ntuple(j -> j == i ? x : s[j], Val(N)))