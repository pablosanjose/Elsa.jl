#######################################################################
# SystemInfo
#######################################################################

struct SystemInfo{Tv,S}
    sampledterms::S
    namesdict::Dict{NameType,Int}
    names::Vector{NameType}
    nsites::Vector{Int}
    norbitals::Vector{Int}
    dims::Vector{Int}       # Hamiltonian block dimensions for each sublattice
    offsets::Vector{Int}    # Hamiltonian block offset for each sublattice
end

SystemInfo{Tv}(terms::S, args...) where {Tv,S} = SystemInfo{Tv,S}(terms, args...)

function SystemInfo(lat::Lattice{E,L,T}, model::Model{Tv}, prevsamples...) where {E,L,T,Tv}
    ns = length(lat.sublats)
    nsites = Vector{Int}(undef, ns)
    namesdict = Dict{NameType,Int}()
    names = Vector{NameType}(undef, ns)
    zeropos = zero(SVector{E,T})
    for (i, sublat) in enumerate(lat.sublats)
        nsites[i] = length(sublat.sites)
        namesdict[sublat.name] = i
        names[i] = sublat.name
    end
    sampledterms = getsamples(namesdict, zeropos, model.terms...)
    allterms = (prevsamples..., sampledterms...)
    norbitals = zeros(Int, ns)
    _fillorbitals!(norbitals, allterms...)
    dims = zeros(Int, ns)
    offsets = zeros(Int, ns + 1)
    for i in eachindex(lat.sublats)
        dims[i] = norbitals[i] * nsites[i]
        offsets[i + 1] = offsets[i] + dims[i]
    end
    return SystemInfo{Tv}(allterms, namesdict, names, nsites, norbitals, dims, offsets)
end

getsamples(namesdict, zeropos) = ()
getsamples(namesdict, zeropos, term, terms...) = 
    (_getsamples(namesdict, zeropos, term), getsamples(namesdict, zeropos, terms...)...)

_getsamples(namesdict, zeropos, term) = 
    (term, term(zeropos, zeropos), _getsublats(namesdict, term.sublats))

_getsublats(namesdict, s::Missing) = s
_getsublats(namesdict, s::NTuple{N,Any}) where {N} = 
    ntuple(i -> (sublatindex(namesdict, first(s[i])), 
                 sublatindex(namesdict, last(s[i]))), Val(N))

_fillorbitals!(norbitals) = nothing

function _fillorbitals!(norbitals, sample::Tuple, samples...) 
    _fillorbitals!(norbitals, sample...) 
    _fillorbitals!(norbitals, samples...)
end

function _fillorbitals!(norbitals, term, ::SMatrix{N,M}, ::Missing) where {N,M} 
    N == M ? fill!(norbitals, N) : throw(DimensionMismatch("Inconsitent model orbital dimensions"))
    return nothing
end

_fillorbitals!(norbitals, term, sm::SMatrix, ss::NTuple{N,Tuple{Int,Int}}) where {N} = 
    foreach(s -> _fillorbitals!(norbitals, sm, s), ss)
_fillorbitals!(norbitals, ::SMatrix{N,M}, (s1, s2)::Tuple{Int,Int}) where {N,M} = 
    (_fillorbitals!(norbitals, Val(N), s1); _fillorbitals!(norbitals, Val(M), s2))

function _fillorbitals!(norbitals, ::Val{N}, s) where {N}
    0 < s <= length(norbitals) || return nothing
    if norbitals[s] == 0
        norbitals[s] = N
    elseif norbitals[s] != N 
        throw(DimensionMismatch("Inconsitent model orbital dimensions"))
    end
    return nothing
end

function tosite(row, sysinfo)
    s = findsublat(row, sysinfo.offsets)
    offset = sysinfo.offsets[s]
    norbs = sysinfo.norbitals[s]
    deltaoffset = row - offset - 1
    site = div(deltaoffset, norbs) + 1
    orboffset = rem(deltaoffset, norbs) + 1
    return site, orboffset, s
end

torow(siteindex, sublat, sysinfo) = 
    sysinfo.offsets[sublat] + (siteindex - 1) * sysinfo.norbitals[sublat] + 1

function findsublat(row, offsets)
    for n in eachindex(offsets)
        offsets[n] >= row && return n - 1
    end
    return 0
end

#######################################################################
# Operator
#######################################################################

mutable struct Block{Tv,L}
    ndist::SVector{L,Int}
    matrix::SparseMatrixCSC{Tv,Int}
    nlinks::Int
end

function Block(ndist, matrix, sysinfo::SystemInfo)
    b = Block(ndist, matrix, 0)
    isempty(matrix) || updatenlinks!(b, sysinfo)
    return b
end

Base.isempty(b::Block) = isempty(b.matrix)

Base.zero(b::Block{Tv,L}) where {Tv,L} = 
    Block(zero(SVector{L,Int}), spzeros(Tv, size(b.matrix)...), 0)

function Base.show(io::IO, b::Block{Tv,L}) where {Tv,L}
    print(io, "Block{$Tv,$L}: Bloch harmonic $(b.ndist) of dimensions $(size(b.matrix)) with $(nnz(b.matrix)) elements")
end

struct Operator{Tv,L}
    matrix::SparseMatrixCSC{Tv,Int}
    intra::Block{Tv,L}
    inters::Vector{Block{Tv,L}}
    boundary::Vector{Tuple{Int,Int}} # ptrs to matrix and intra that contain terms in any inters
end

function Operator{Tv,L}(n::Int) where {Tv,L} 
    return Operator(
        spzeros(n,n), 
        Block{Tv,L}(zero(SVector{L,Int}), n),  
        Block{Tv,L}[],
        Tuple{Int,Int}[])
end

function Base.show(io::IO, op::Operator{L,Tv}) where {L,Tv}
    print(io, "Operator{$L,$Tv}: Bloch operator of dimensions $(size(op.matrix)) with $(nnz(op.matrix)) elements in main matrix")
end

#######################################################################
# Operator API
#######################################################################

nlinks(o::Operator) = nlinks(o.intra) + (isempty(o.inters) ? 0 : sum(nlinks, o.inters))
nlinks(b::Block) = b.nlinks

nsublats(b::Block) = nsublats(b.sysinfo)

function updatenlinks!(b::Block, sysinfo) 
    n = 0
    zeron = iszero(b.ndist)
    for (_, (target, source), (row, col), _) in BlockIterator(b, sysinfo)
        (!zeron || row != col) && (n += 1)
    end
    b.nlinks = n
    return nothing
end

insertblochphases!(o::Operator{Tv,L}, kn) where {Tv<:AbstractFloat,L} = 
    throw(DomainError(Tv, "Cannot apply Bloch phases to a real Hamiltonian."))

function insertblochphases!(op::Operator{Tv,L}, ϕn, dϕaxis = missing) where {Tv,L} 
    length(ϕn) == L || throw(DimensionMismatch(
        "The dimension of the normalized Bloch phases should match the lattice dimension $L"))
    rows = rowvals(op.matrix)
    vals = nonzeros(op.matrix)
    valsintra = nonzeros(op.intra.matrix)
    for (ptrmatrix, ptrintra) in op.boundary
        vals[ptrmatrix] = iszero(ptrintra) ? zero(Tv) : valsintra[ptrintra]
    end
    for inter in op.inters
        # dϕaxis != missing produces derivatives of Bloch phases respect a given axis
        dexp = dϕaxis === missing ? 1.0 : 2pi * im * dot(inter.ndist, dϕaxis)
        phase = dexp * exp(2pi * im * dot(inter.ndist, ϕn))
        for (i,j,v,ptr) in SparseMatrixReader(inter.matrix)
            for ptr in nzrange(op.matrix,j)
                rows[ptr] == i && (vals[ptr] += phase * v; break)
            end
        end
    end
    return op
end

function boundaryoperator(op::Operator{Tv}) where {Tv}
    n = length(op.boundary)
    sb = SparseMatrixBuilder{Tv}(size(op.matrix)...)
    rows = rowvals(op.matrix)
    vals = nonzeros(op.matrix)
    sofar = 1
    for col in 1:size(op.matrix, 2)
        colrange = nzrange(op.matrix, col)
        for bidx in sofar:n
            ptr = first(op.boundary[bidx])
            if ptr in colrange
                sofar = bidx + 1
                pushtocolumn!(sb, rows[ptr], vals[ptr])
            else
                break
            end
        end
        finalizecolumn!(sb)
    end
    matrix = sparse(sb)
    intra = zero(op.intra)
    inters = op.inters
    boundary = extractboundary(matrix, intra, inters)
    return Operator(matrix, intra, inters, boundary)
end
