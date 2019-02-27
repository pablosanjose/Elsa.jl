#######################################################################
# SystemInfo
#######################################################################
struct SystemInfo{Tv,S}
    sampledterms::S
    namesdict::Dict{NameType,Int}
    names::Vector{NameType}
    nsites::Vector{Int}
    norbitals::Vector{Int}
    dims::Vector{Int}
    offsets::Vector{Int}    
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

sublatindex(s::SystemInfo, name) = sublatindex(s.namesdict, name)
sublatindex(d::Dict, name::NameType) = d[name]
sublatindex(s::Dict, i::Integer) = Int(i)
# Base.keys(s::SystemInfo) = keys(s.nsites)

function tosite(row, sysinfo)
    s = findsublat(row, sysinfo.offsets)
    offset = sysinfo.offsets[s]
    norbs = sysinfo.norbitals[s]
    delta = row - offset
    return div(delta, norbs), rem(delta, norbs), s
end

torow(siteindex, s, sysinfo) = 
    sysinfo.offsets[s] + (siteindex - 1) * sysinfo.norbitals[s] + 1

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

insertblochphases!(o::Operator{Tv}, kn) where {Tv<:AbstractFloat} = 
    throw(DomainError(Tv, "Cannot apply Bloch phases to a real Hamiltonian."))
function insertblochphases!(op::Operator{Tv}, kn) where {Tv}
    rows = rowvals(op.matrix)
    vals = nonzeros(op.matrix)
    valsintra = nonzeros(op.intra.matrix)
    for (ptrmatrix, ptrintra) in op.boundary
        vals[ptrmatrix] = iszero(ptrintra) ? zero(Tv) : valsintra[ptrintra]
    end
    for inter in op.inters
        phase = exp(2pi * im * dot(inter.ndist, kn))
        for (i,j,v,ptr) in SparseMatrixReader(inter.matrix)
            for ptr in nzrange(op.matrix,j)
                rows[ptr] == i && (vals[ptr] += phase * v; break)
            end
        end
    end
    return op
end

#######################################################################
# Fields
#######################################################################
# struct Field{F,S}
#     f::F
#     sublats::S              # SL === Missing means any sublats 
# end

# field(f, sublats...) = Field(f, _normaliseSLpairs(sublats))
# field(f) = Field(f, missing)
# (f::Field)(s::S, r, dr) where {S<:SMatrix} = ensureSMatrix(f.f(s,r,dr))


# #######################################################################
# # BlochVector
# #######################################################################

# struct BlochVector{Tv,L}
#     I::Vector{Int}
#     J::Vector{Int}
#     V::Vector{Tv}
#     Voffsets::Vector{Int}
#     Vns::NTuple{L, Vector{Vector{Tv}}}
#     ndists::Vector{SVector{L, Int}}
#     sublatorbitals::Vector{Int}   # orbitals in each sublattice
#     sublatoffsets::Vector{Int}    # first index in each sublattice block
#     workspace::SparseWorkspace{Tv}
#     matrix::SparseMatrixCSC{Tv,Int}
# end

# function BlochVector{Tv,L}() where {Tv,L} 
#     I = Int[]
#     J = Int[]
#     V = Tv[]
#     Vns = ntuple(_ -> Vector{Tv}[], Val(L))
#     Voffsets = Int[]
#     ndists = SVector{L,Int}[]
#     sublatorbitals = Int[]
#     sublatoffsets = Int[]
#     workspace = SparseWorkspace{Tv}(0, 0)
#     mat = sparse(fill(zero(Tv), (0, 0)))
#     return BlochVector(I, J, V, Voffsets, Vns, ndists, sublatorbitals, sublatoffsets, workspace, mat)
# end

# function Base.show(io::IO, op::BlochVector{Tv,L}) where {Tv,L}
#     print(io, "Bloch $L-vector of dimensions $(size(op.matrix)) with $(nnz(op.matrix)) elements")
# end

# function insertblochphases!(bvec::BlochVector{Tv,L}, kn, axis) where {Tv,L}
# 	L > 0 && _insertblochphases!(bvec.V, bvec.Vns[axis], bvec.Voffsets, bvec.ndists, convert(SVector{L,Tv}, kn), false)
# 	return nothing
# end

# function gradient(op::Operator{Tv,L}; kn::SVector{L,Int} = zero(SVector{L,Int}), axis::Int = 1) where {Tv,L}
#     ndists = op.ndists
# 	workspace = op.workspace
# 	dim = size(op.matrix, 1)
# 	offset = op.Voffsets[1]
# 	I = op.I[offset:end]
# 	J = op.J[offset:end]
# 	V = zeros(Tv, length(I))
# 	Voffsets = op.Voffsets .- offset .+ 1
# 	Vns = ntuple(ax -> [(2pi * im * n[ax]) .* v for (n, v) in zip(ndists, op.Vn)], Val(L))
# 	L > 0 && _insertblochphases!(V, Vns[axis], Voffsets, ndists, kn, false)
#     matrix = sparse!(I, J, V, dim, workspace)
#     sublatorbitals = op.sublatorbitals
#     sublatoffsets = op.sublatoffsets
# 	return BlochVector(I, J, V, Voffsets, Vns, ndists, sublatorbitals, sublatoffsets, workspace, matrix)
# end

# _insertblochphases!(V::AbstractArray{T}, _...) where {T<:AbstractFloat} = throw(DomainError(T, "Cannot apply Bloch phases to a real Hamiltonian."))
# function _insertblochphases!(V::AbstractArray{Complex{T}}, Vn, Voffsets, ndists, kn, intracell) where {T}
#     if intracell
#         V[Voffsets[1]:end] .= zero(Complex{T})
#     else
#         for n in 1:(length(Voffsets) - 1)
#             phase = exp(2pi * im * dot(ndists[n], kn))
#             for (Vnj, Vj) in enumerate(Voffsets[n]:(Voffsets[n + 1] - 1))
#                 V[Vj] = Vn[n][Vnj] * phase
#             end
#         end
#     end
#     return nothing
# end
