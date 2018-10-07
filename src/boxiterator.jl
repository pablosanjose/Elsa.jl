struct BoxRegister{N}
    cellinds::Vector{Tuple{NTuple{N,Int}, Int}}
end

BoxRegister{N}() where N = BoxRegister(Tuple{NTuple{N,Int}, Int}[])

"""
    BoxIterator(seed::NTuple{N,Int}; maxiterations = missing, nregisters = 0)

Cartesian iterator `iter` over N-tuples of integers (`cell`s) that starts at `seed` and
grows outwards in the form of a box of increasing sides (not necesarily equal) until it 
encompasses a certain N-dimensional region. The field `iter.monitor::Bool` is used by 
the user at each iteration to signal whether the current `cell` is inside the region. 
The option `nregisters = n` creates `n` `BoxRegister`s that store `(cell, index)`
"""
struct BoxIterator{N}
    seed::NTuple{N,Int}
    maxiter::Union{Int, Missing}
    dimdir::MVector{2,Int}
    nmoves::MVector{N,Bool}
    pmoves::MVector{N,Bool}
    npos::MVector{N,Int}
    ppos::MVector{N,Int}
    registers::Vector{BoxRegister{N}}
end

IteratorSize(::BoxIterator) = SizeUnknown()
IteratorEltype(::BoxIterator) = HasElType()
eltype(::BoxIterator{N}) where {N} = NTuple{N,Int}
boundingboxiter(b::BoxIterator) = (b.npos, b.ppos)

function BoxIterator(seed::NTuple{N}; maxiterations::Union{Int,Missing} = missing, nregisters::Int = 0) where {N} 
    BoxIterator(seed, maxiterations, MVector(1, 2),
        ones(MVector{N,Bool}), ones(MVector{N,Bool}), MVector(seed), MVector(seed),
        nregisters > 0 ? [BoxRegister{N}() for i in 1:nregisters] : BoxRegister{N}[])
end

function iteratorreset!(b::BoxIterator{N}) where {N}
    b.dimdir[1] = 1
    b.dimdir[2] = 2
    b.nmoves .= ones(MVector{N,Bool})
    b.pmoves .= ones(MVector{N,Bool})
    b.npos   .= MVector(b.seed)
    b.ppos   .= MVector(b.seed)
    nregisters = length(b.registers)
    nregisters > 0 && (b.registers .= [BoxRegister{N}() for i in 1:nregisters])
    return nothing
end

struct BoxIteratorState{N}
    range::CartesianIndices{N, NTuple{N,UnitRange{Int}}}
    rangestate::CartesianIndex{N}
    iteration::Int
end

function Base.iterate(b::BoxIterator{N}) where {N}
    range = CartesianIndices(ntuple(i -> b.seed[i]:b.seed[i], Val(N)))
    itrange = iterate(range)
    if itrange === nothing
        return nothing
    else
        (cell, rangestate) = itrange
        return (cell.I, BoxIteratorState(range, rangestate, 1))
    end
end

function Base.iterate(b::BoxIterator{N}, s::BoxIteratorState{N}) where {N}
    itrange = iterate(s.range, s.rangestate)
    facedone = itrange === nothing
    if facedone
        alldone = !any(b.pmoves) && !any(b.nmoves) || isless(b.maxiter, s.iteration)
        if alldone  # Last shells in all directions were empty, trim from boundingboxiter
            b.npos .+= 1
            b.ppos .-= 1
            return nothing
        else
            newrange = nextface!(b)
            # newrange === nothing && return nothing
            itrange = iterate(newrange)
            # itrange === nothing && return nothing
            (cell, rangestate) = itrange
            return (cell.I, BoxIteratorState(newrange, rangestate, s.iteration + 1))
        end
    else
        (cell, rangestate) = itrange
        return (cell.I, BoxIteratorState(s.range, rangestate, s.iteration + 1))
    end
end

function nextface!(b::BoxIterator{N}) where {N}
    @inbounds for i in 1:2N
        nextdimdir!(b)
        newdim, newdir = Tuple(b.dimdir)
        if newdir == 1
            if b.nmoves[newdim]
                b.npos[newdim] -= 1
                b.nmoves[newdim] = false
                return newrangeneg(b, newdim)
            end
        else
            if b.pmoves[newdim]
                b.ppos[newdim] += 1
                b.pmoves[newdim] = false
                return newrangepos(b, newdim)
            end
        end
    end
    return nothing
end

@inline function nextdimdir!(b::BoxIterator{N}) where {N}
    dim, dir = Tuple(b.dimdir)
    if dim < N
        dim += 1
    else
        dim = 1
        dir = ifelse(dir == 1, 2, 1)
    end
    b.dimdir[1] = dim
    b.dimdir[2] = dir
    return nothing
end

function newrangeneg(b::BoxIterator{N}, dim) where {N}
    return CartesianIndices(ntuple(
        i -> ifelse(i == dim, b.npos[i]:b.npos[i], b.npos[i]:b.ppos[i]),
        Val(N)))
end

function newrangepos(b::BoxIterator{N}, dim) where {N}
    return CartesianIndices(ntuple(
        i -> ifelse(i == dim, b.ppos[i]:b.ppos[i], b.npos[i]:b.ppos[i]),
        Val(N)))
end

function acceptcell!(b::BoxIterator{N}, cell) where {N}
    dim, dir = Tuple(b.dimdir)
    if dir == 1
        @inbounds for i in 1:N
            (cell[i] == b.ppos[i]) && (b.pmoves[i] = true)
            (i == dim || cell[i] == b.npos[i]) && (b.nmoves[i] = true)
        end
    else
        @inbounds for i in 1:N
            (i == dim || cell[i] == b.ppos[i]) && (b.pmoves[i] = true)
            (cell[i] == b.npos[i]) && (b.nmoves[i] = true)
        end
    end
    return nothing
end

function registersite!(iter, cell, sublat, idx)
    push!(iter.registers[sublat].cellinds, (cell, idx))
    return nothing
end