#######################################################################
# BoxIterator
#######################################################################

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

Base.IteratorSize(::BoxIterator) = Base.SizeUnknown()
Base.IteratorEltype(::BoxIterator) = Base.HasEltype()
Base.eltype(::BoxIterator{N}) where {N} = NTuple{N,Int}
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

#######################################################################
# NeighborIterator
#######################################################################

mutable struct NeighborIterator{L}
    l::L
    src::Int
    s1::Int
    s2::Int
end
Base.IteratorSize(::NeighborIterator) = Base.HasLength()
Base.IteratorEltype(::NeighborIterator) = Base.HasEltype()
Base.eltype(::NeighborIterator) = Int
Base.length(ni::NeighborIterator{<:Ilink}) = numneighbors(ni, 0)
function Base.length(ni::NeighborIterator{<:Links})
    l = numneighbors(ni, 0)
    for nilink in 1:ninterlinks(ni.l)
        l += numneighbors(ni, nilink)
    end
    return l
end
numneighbors(ni::NeighborIterator, nlink) = length(nzrange(slink(ni, nlink).rdr, ni.src))

slink(ni::NeighborIterator{<:Links}, nilink) = iszero(nilink) ? ni.l.intralink.slinks[ni.s2, ni.s1] : ni.l.interlinks[nilink].slinks[ni.s2, ni.s1]
slink(ni::NeighborIterator{<:Ilink}, nilink) = ni.l.slinks[ni.s2, ni.s1]
maxilinkindex(ni::NeighborIterator{<:Links}) = ninterlinks(ni.l)
maxilinkindex(ni::NeighborIterator{<:Ilink}) = 0

function iterate(ni::NeighborIterator{<:Links}, state = (0, 1))
    (nilink, ptr) = state
    nilink > maxilinkindex(ni) && return nothing
    s = slink(ni, nilink)
    range = nzrange(s.rdr, ni.src)
    ptr > length(range) && return iterate(ni, (nilink + 1, 1))
    targets = rowvals(s.rdr)
    return (targets[range[ptr]], (nilink, ptr + 1))
end
function iterate(ni::NeighborIterator{<:Ilink}, state = (0, 1, slink(ni, 0)))
    (nilink, ptr, s) = state
    range = nzrange(s.rdr, ni.src)
    targets = rowvals(s.rdr)
    ptr > length(range) ? nothing : @inbounds (targets[range[ptr]], (nilink, ptr + 1, s))
end

NeighborIterator(l::Links, src, sublats, onlyintra::Val{true}) =  NeighborIterator(l.intralink, src, sublats)
NeighborIterator(l::Links, src, sublats, onlyintra::Val{false}) =  NeighborIterator(l, src, sublats)
NeighborIterator(l, src, (s1,s2)::Tuple{Int,Int}) = NeighborIterator(l, src, s1, s2)

neighbors!(ni::NeighborIterator, src) = (ni.src = src; return ni)
neighbors(p...) = NeighborIterator(p...)

neighbors_rdr(s::Slink, src) = ((rowvals(s.rdr)[j], nonzeros(s.rdr)[j]) for j in nzrange(s.rdr, src))
neighbors_rdr(s::Slink) = zip(s.rdr.rowval, s.rdr.nzval)

#######################################################################
# SparseMatrixBuilder
#######################################################################

mutable struct SparseMatrixBuilder{T}
    m::Int
    n::Int
    colptr::Vector{Int}
    rowval::Vector{Int}
    nzval::Vector{T}
    colcounter::Int
    rowvalcounter::Int
end

function SparseMatrixBuilder(::Type{T}, m, n, coordinationhint = 1) where T
    colptr = Vector{Int}(undef, n + 1)
    colptr[1] = 1
    rowval = Int[]; sizehint!(rowval, round(Int, 0.5 * coordinationhint * n))
    nzval = T[];    sizehint!(nzval,  round(Int, 0.5 * coordinationhint * n))
    # The 0.5 is due to storing undirected links only
    return SparseMatrixBuilder(m, n, colptr, rowval, nzval, 1, 1)
end

function pushtocolumn!(s::SparseMatrixBuilder, row, x, skipdupcheck = true)
    if skipdupcheck || !isintail(row, s.rowval, s.colptr[s.colcounter])
        push!(s.rowval, row)
        push!(s.nzval, x)
        s.rowvalcounter += 1
    end
    return x
end

function isintail(element, container, start::Int)
    for i in start:length(container)
        container[i] == element && return true
    end
    return false
end

function finalisecolumn!(s::SparseMatrixBuilder)
    s.colcounter > s.n && throw(DimensionMismatch("Pushed too many columns to matrix"))
    s.colcounter += 1
    s.colptr[s.colcounter] = s.rowvalcounter
    return nothing
end

function SparseArrays.sparse(s::SparseMatrixBuilder)
    if s.colcounter < s.n + 1
        for col in (s.colcounter + 1):(s.n + 1)
            s.colptr[col] = s.rowvalcounter
        end
    end
    return SparseMatrixCSC(s.m, s.n, s.colptr, s.rowval, s.nzval)
end

#######################################################################
# SparseMatrixReader
#######################################################################

struct SparseMatrixReader{T,TI}
    matrix::SparseMatrixCSC{T,TI}
end

Base.IteratorSize(::SparseMatrixReader) = Base.HasLength()
Base.IteratorEltype(::SparseMatrixReader) = Base.HasEltype()
Base.eltype(::SparseMatrixReader{T,TI}) where {T,TI} = Tuple{TI,TI,T}
Base.length(s::SparseMatrixReader) = nnz(s.matrix)

function iterate(s::SparseMatrixReader, state = (1, 1))
    (ptr, col) = state
    ptr > length(s) && return nothing
    @inbounds while ptr > s.matrix.colptr[col + 1] - 1
         col += 1
    end
    return (s.matrix.rowval[ptr], col, s.matrix.nzval[ptr], ptr), (ptr + 1, col)
end

enumerate_sparse(s::SparseMatrixCSC) = SparseMatrixReader(s)
