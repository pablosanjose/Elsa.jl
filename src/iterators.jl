#######################################################################
# BoxIterator
#######################################################################
struct BoxRegister{N}
    cellinds::Vector{Tuple{SVector{N,Int}, Int}}
end

BoxRegister{N}() where N = BoxRegister(Tuple{NTuple{N,Int}, Int}[])

"""
    BoxIterator(seed::SVector{N,Int}; maxiterations = missing, nregisters = 0)

Cartesian iterator `iter` over `SVector{N,Int}`s (`cell`s) that starts at `seed` and
grows outwards in the form of a box of increasing sides (not necesarily equal) until 
it encompasses a certain N-dimensional region. To signal that a cell is in the desired 
region the user calls `acceptcell!(iter, cell)`.  The option `nregisters = n` creates `n` 
`BoxRegister`s that store `(cell, index)`
"""
struct BoxIterator{N}
    seed::SVector{N,Int}
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
Base.eltype(::BoxIterator{N}) where {N} = SVector{N,Int}
boundingboxiter(b::BoxIterator) = (Tuple(b.npos), Tuple(b.ppos))

function BoxIterator(seed::SVector{N}; maxiterations::Union{Int,Missing} = missing, nregisters::Int = 0) where {N}
    BoxIterator(seed, maxiterations, MVector(1, 2),
        ones(MVector{N,Bool}), ones(MVector{N,Bool}), MVector{N,Int}(seed), MVector{N,Int}(seed),
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

Base.iterate(b::BoxIterator{0}) = (SVector{0,Int}(), nothing)
Base.iterate(b::BoxIterator{0}, state) = nothing

function Base.iterate(b::BoxIterator{N}) where {N}
    range = CartesianIndices(ntuple(i -> b.seed[i]:b.seed[i], Val(N)))
    itrange = iterate(range)
    if itrange === nothing
        return nothing
    else
        (cell, rangestate) = itrange
        return (SVector(cell.I), BoxIteratorState(range, rangestate, 1))
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
            return (SVector(cell.I), BoxIteratorState(newrange, rangestate, s.iteration + 1))
        end
    else
        (cell, rangestate) = itrange
        return (SVector(cell.I), BoxIteratorState(s.range, rangestate, s.iteration + 1))
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
# Fallback for non-BoxIterators
acceptcell!(b, cell) = nothing 

function registersite!(iter, cell, sublat, idx)
    push!(iter.registers[sublat].cellinds, (cell, idx))
    return nothing
end

#######################################################################
# CoSort
#######################################################################
struct CoSortTup{T,Tv}
    x::T
    y::Tv
end

mutable struct CoSort{T,Tv,S<:AbstractVector{T},C<:AbstractVector{Tv}} <: AbstractVector{CoSortTup{T,Tv}}
    sortvector::S
    covector::C
    offset::Int
    function CoSort{T,Tv,S,C}(sortvector, covector, offset) where {T,Tv,S,C}
        length(covector) >= length(sortvector) ? new(sortvector, covector, offset) :
            throw(DimensionMismatch("Coarray length should exceed sorting array"))
    end
end
CoSort(sortvector::S, covector::C) where {T,Tv,S<:AbstractVector{T},C<:AbstractVector{Tv}} = 
    CoSort{T,Tv,S,C}(sortvector, covector, 0)

Base.size(c::CoSort) = (size(c.sortvector, 1) - c.offset,)
Base.getindex(c::CoSort, i) = CoSortTup(getindex(c.sortvector, i + c.offset), getindex(c.covector, i + c.offset))
Base.setindex!(c::CoSort, t::CoSortTup, i) = (setindex!(c.sortvector, t.x, i + c.offset); setindex!(c.covector, t.y, i + c.offset); c) 
Base.isless(a::CoSortTup, b::CoSortTup) = isless(a.x, b.x)
Base.Sort.defalg(v::C) where {T<:Union{Number, Missing}, C<:CoSort{T}} = Base.DEFAULT_UNSTABLE

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
    cosorter::CoSort{Int,T,Vector{Int},Vector{T}}
end

function SparseMatrixBuilder{Tv}(m, n, coordinationguess = 0) where Tv
    colptr = Vector{Int}(undef, n + 1)
    colptr[1] = 1
    rowval = Int[]
    nzval = Tv[]
    if !iszero(coordinationguess)
        sizehint!(rowval, 2 * coordinationguess * m)
        sizehint!(nzval, 2 * coordinationguess * m)
    end
    return SparseMatrixBuilder(m, n, colptr, rowval, nzval, 1, 1, CoSort(rowval, nzval))
end
SparseArrays.nzrange(S::SparseMatrixBuilder, col::Integer) = 
    S.colptr[col]:(S.colptr[col+1]-1)
SparseArrays.rowvals(S::SparseMatrixBuilder) = S.rowval
SparseArrays.nonzeros(S::SparseMatrixBuilder) = S.nzval
Base.size(S::SparseMatrixBuilder) = (S.n, S.m)
Base.size(S::SparseMatrixBuilder, k) = size(S)[k]

function pushtocolumn!(s::SparseMatrixBuilder, row::Int, x, skipdupcheck::Bool = true)
    if skipdupcheck || !isintail(row, s.rowval, s.colptr[s.colcounter])
        push!(s.rowval, row)
        push!(s.nzval, x)
        s.rowvalcounter += 1
    end
    return s
end

# pushtocolumn!(s::SparseMatrixBuilder, rows::AbstractArray, xs::AbstractArray) = 
#     pushtocolumn!(s, rows, xs, eachindex(rows))
# function pushtocolumn!(s::SparseMatrixBuilder, rows::AbstractArray, xs::AbstractArray, range)
#     n = length(range)
#     Base._growend!(s.rowval, n)
#     copyto!(s.rowval, length(s.rowval)-n+1, rows, first(range), n)
#     Base._growend!(s.nzval, n)
#     copyto!(s.nzval, length(s.nzval)-n+1, xs, first(range), n)
#     s.rowvalcounter += n
#     return s
# end

function isintail(element, container, start::Int)
    for i in start:length(container)
        container[i] == element && return true
    end
    return false
end

function finalisecolumn!(s::SparseMatrixBuilder, sortcol::Bool = false)
    s.colcounter > s.n && throw(DimensionMismatch("Pushed too many columns to matrix"))
    !sortcol && partialsort!(s.cosorter, s.colptr[s.colcounter]:s.rowvalcounter)
    s.colcounter += 1
    s.colptr[s.colcounter] = s.rowvalcounter
    return nothing
end

function finalisecolumn!(s::SparseMatrixBuilder, ncols::Int)
    for _ in 1:ncols
        finalisecolumn!(s)
    end
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

struct SparseMatrixReader{Tv,Ti}
    matrix::SparseMatrixCSC{Tv,Ti}
end

Base.IteratorSize(::SparseMatrixReader) = Base.HasLength()
Base.IteratorEltype(::SparseMatrixReader) = Base.HasEltype()
Base.eltype(::SparseMatrixReader{Tv,Ti}) where {Tv,Ti} = Tuple{Ti,Ti,Tv}
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

#######################################################################
# BlockIterator
#######################################################################
struct BlockIterator{B<:Block,S<:SystemInfo}
    block::B
    sysinfo::S
end

# returns ((s1, s2), (target, source), (row, col), boxsize, ptr), one per box (site pair), 
# where boxsize is (N,M) for orbitals in site pair
function Base.iterate(blockiter::BlockIterator, state = (1, nextcolumn(blockiter)))
    block = blockiter.block
    sysinfo = blockiter.sysinfo
    (ptr, col) = state
    if col === 0 || !checkbounds(Bool, rowvals(block.matrix), ptr)
        return nothing
    else
        row = rowvals(block.matrix)[ptr]
        s1 = findsublat(row, sysinfo.offsets)
        s2 = findsublat(col, sysinfo.offsets)
        (iszero(s1) || iszero(s2)) && throw(
            ErrorException("Unexpected row/col ($row, $col) out of offset range"))
        (n, m) = sysinfo.norbitals[s1], sysinfo.norbitals[s2]
        rangecol = nzrange(block.matrix, col)
        newptr = ptr + n
        if newptr > maximum(rangecol) 
            newcol = nextcolumn(blockiter, col)
            newptr += length(rangecol) * (m - 1)
        else
            newcol = col
        end
        targetsite = 1 + (row - 1 - sysinfo.offsets[s1]) ÷ n
        sourcesite = 1 + (col - 1 - sysinfo.offsets[s2]) ÷ m
        return ((s1, s2), (targetsite, sourcesite), (row, col), (n, m), ptr), 
               (newptr, newcol)
    end
end
nextcolumn(blockiter) = 
    isempty(nzrange(blockiter.block.matrix, 1)) ? nextcolumn(blockiter, 1) : 1
function nextcolumn(blockiter, col)
    c = col
    while true
        s = findsublat(c, blockiter.sysinfo.offsets)
        iszero(s) && break
        c += blockiter.sysinfo.norbitals[s]
        c > size(blockiter.block.matrix, 2) && break
        isempty(nzrange(blockiter.block.matrix, c)) || return c
    end
    return 0
end
Base.IteratorSize(::BlockIterator) = Base.SizeUnknown()
Base.IteratorEltype(::BlockIterator) = Base.HasEltype()
Base.eltype(::BlockIterator) = 
    Tuple{Tuple{Int,Int},Tuple{Int,Int},Tuple{Int,Int},Tuple{Int,Int},Int}