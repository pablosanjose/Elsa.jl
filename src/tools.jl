toSMatrix() = SMatrix{0,0,Float64}()
toSMatrix(ss::NTuple{N,Number}...) where {N} = toSMatrix(SVector{N}.(ss)...)
toSMatrix(ss::SVector{N}...) where {N} = hcat(ss...)
toSMatrix(::Type{T}, ss...) where {T} = _toSMatrix(T, toSMatrix(ss...))
_toSMatrix(::Type{T}, s::SMatrix{N,M}) where {N,M,T} = convert(SMatrix{N,M,T}, s)
# Dynamic dispatch
toSMatrix(ss::AbstractVector...) = toSMatrix(Tuple.(ss)...)
toSMatrix(s::AbstractMatrix) = SMatrix{size(s,1), size(s,2)}(s)

toSVectors(vs...) = [promote(toSVector.(vs)...)...]
toSVector(v::SVector) = v
toSVector(v::NTuple{N,Number}) where {N} = SVector(v)
toSVector(x::Number) = SVector{1}(x)
toSVector(::Tuple{}) = SVector{0,Float64}()
toSVector(::Type{T}, v) where {T} = T.(toSVector(v))
toSVector(::Type{T}, ::Tuple{}) where {T} = SVector{0,T}()
# Dynamic dispatch
toSVector(v::AbstractVector) = SVector(Tuple(v))

# ensureSMatrix(f::Function) = f
# ensureSMatrix(m::T) where T<:Number = SMatrix{1,1,T,1}(m)
# ensureSMatrix(m::SMatrix) = m
# ensureSMatrix(m::Array) =
#     throw(ErrorException("Write all model terms using scalars or @SMatrix[matrix]"))

_rdr(r1, r2) = (0.5 * (r1 + r2), r2 - r1)

zerotuple(::Type{T}, ::Val{L}) where {T,L} = ntuple(_ -> zero(T), Val(L))

padright(sv::StaticVector{E,T}, x::T, ::Val{E}) where {E,T} = sv
padright(sv::StaticVector{E,T}, x::T2, ::Val{E2}) where {E,T,E2,T2} =
    SVector{E2, T2}(ntuple(i -> i > E ? x : T2(sv[i]), Val(E2)))
padright(sv::StaticVector{E,T}, ::Val{E2}) where {E,T,E2} = padright(sv, zero(T), Val(E2))

@inline pad(s::SMatrix{E,L}, st::Type{S}) where {E,L,E2,L2,T2,S<:SMatrix{E2,L2,T2}} =
    S(ntuple(k -> _pad((k - 1) % E2 + 1, (k - 1) ÷ E2 + 1, zero(T2), s), Val(E2 * L2)))

@inline _pad(i, j, zero, s::SMatrix{E,L}) where {E,L} =
    i > E || j > L ? zero : s[i,j]

## Work around BUG: -SVector{0,Int}() isa SVector{0,Union{}}
negative(s::SVector{L,<:Number}) where {L} = -s
negative(s::SVector{0,<:Number}) where {L} = s

function nnzdiag(s::SparseMatrixCSC)
    count = 0
    rowptrs = rowvals(s)
    for col in 1:size(s,2)
        for ptr in nzrange(s, col)
            rowptrs[ptr] == col && (count += 1; break)
        end
    end
    return count
end
nnzdiag(s::Matrix) = minimum(size(s))

pinverse(s::SMatrix) = (qrfact = qr(s); return inv(qrfact.R) * qrfact.Q')

display_as_tuple(v, prefix = "") = isempty(v) ? "()" : 
    string("(", prefix, join(v, string(", ", prefix)), ")")

# padrightbottom(m::Matrix{T}, im, jm) where {T} = padrightbottom(m, zero(T), im, jm)

# function padrightbottom(m::Matrix{T}, zeroT::T, im, jm) where T
#     i0, j0 = size(m)
#     [i <= i0 && j<= j0 ? m[i,j] : zeroT for i in 1:im, j in 1:jm]
# end

# @inline tuplejoin(x) = x
# @inline tuplejoin(x, y) = (x..., y...)
# @inline tuplejoin(x, y, z...) = (x..., tuplejoin(y, z...)...)
# tuplesort((a,b)::Tuple{<:Number,<:Number}) = a > b ? (b, a) : (a, b)
# tuplesort(t::Tuple) = t
# tuplesort(::Missing) = missing

# collectfirst(s::T, ss...) where {T} = _collectfirst((s,), ss...)
# _collectfirst(ts::NTuple{N,T}, s::T, ss...) where {N,T} = _collectfirst((ts..., s), ss...)
# _collectfirst(ts::Tuple, ss...) = (ts, ss)
# _collectfirst(ts::NTuple{N,System}, s::System, ss...) where {N} = _collectfirst((ts..., s), ss...)
# collectfirsttolast(ss...) = tuplejoin(reverse(collectfirst(ss...))...)



# allorderedpairs(v) = [(i, j) for i in v, j in v if i >= j]

# # Like copyto! but with potentially different tensor orders
# function copyslice!(dest::AbstractArray{T1,N1}, Rdest::CartesianIndices{N1},
#                     src::AbstractArray{T2,N2}, Rsrc::CartesianIndices{N2}) where {T1,T2,N1,N2}
#     isempty(Rdest) && return dest
#     if length(Rdest) != length(Rsrc)
#         throw(ArgumentError("source and destination must have same length (got $(length(Rsrc)) and $(length(Rdest)))"))
#     end
#     checkbounds(dest, first(Rdest))
#     checkbounds(dest, last(Rdest))
#     checkbounds(src, first(Rsrc))
#     checkbounds(src, last(Rsrc))
#     src′ = Base.unalias(dest, src)
#     for (Is, Id) in zip(Rsrc, Rdest)
#         @inbounds dest[Id] = src′[Is]
#     end
#     return dest
# end

######################################################################
# Permutations (taken from Combinatorics.jl)
#######################################################################

struct Permutations{T}
    a::T
    t::Int
end

Base.eltype(::Type{Permutations{T}}) where {T} = Vector{eltype(T)}

Base.length(p::Permutations) = (0 <= p.t <= length(p.a)) ? factorial(length(p.a), length(p.a)-p.t) : 0

"""
    permutations(a)
Generate all permutations of an indexable object `a` in lexicographic order. Because the number of permutations
can be very large, this function returns an iterator object.
Use `collect(permutations(a))` to get an array of all permutations.
"""
permutations(a) = Permutations(a, length(a))

"""
    permutations(a, t)
Generate all size `t` permutations of an indexable object `a`.
"""
function permutations(a, t::Integer)
    if t < 0
        t = length(a) + 1
    end
    Permutations(a, t)
end

function Base.iterate(p::Permutations, s = collect(1:length(p.a)))
    (!isempty(s) && max(s[1], p.t) > length(p.a) || (isempty(s) && p.t > 0)) && return
    nextpermutation(p.a, p.t ,s)
end

function nextpermutation(m, t, state)
    perm = [m[state[i]] for i in 1:t]
    n = length(state)
    if t <= 0
        return(perm, [n+1])
    end
    s = copy(state)
    if t < n
        j = t + 1
        while j <= n &&  s[t] >= s[j]; j+=1; end
    end
    if t < n && j <= n
        s[t], s[j] = s[j], s[t]
    else
        if t < n
            reverse!(s, t+1)
        end
        i = t - 1
        while i>=1 && s[i] >= s[i+1]; i -= 1; end
        if i > 0
            j = n
            while j>i && s[i] >= s[j]; j -= 1; end
            s[i], s[j] = s[j], s[i]
            reverse!(s, i+1)
        else
            s[1] = n+1
        end
    end
    return (perm, s)
end

# Taken from Combinatorics.jl
# TODO: This should really live in Base, otherwise it's type piracy
"""
    factorial(n, k)

Compute ``n!/k!``.
"""
function Base.factorial(n::T, k::T) where T<:Integer
    if k < 0 || n < 0 || k > n
        throw(DomainError((n, k), "n and k must be nonnegative with k ≤ n"))
    end
    f = one(T)
    while n > k
        f = Base.checked_mul(f, n)
        n -= 1
    end
    return f
end

Base.factorial(n::Integer, k::Integer) = factorial(promote(n, k)...)
