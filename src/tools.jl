extended_eps(T) = 10_000*eps(T)
extended_eps() = 10_000*eps(Float64)

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
# Dynamic dispatch
toSVector(v::AbstractVector) = SVector(Tuple(v))

ensureSMatrix(f::Function) = f
ensureSMatrix(m::T) where T<:Number = SMatrix{1,1,T,1}(m)
ensureSMatrix(m::SMatrix) = m
ensureSMatrix(m::Array) = 
    throw(ErrorException("Write all model terms using scalars or @SMatrix[matrix]"))

_rdr(r1, r2) = (0.5 * (r1 + r2), r2 - r1)

zerotuple(::Type{T}, ::Val{L}) where {T,L} = ntuple(_ -> zero(T), Val(L))

padright(sv::StaticVector{E,T}, x::T, ::Val{E}) where {E,T} = sv
padright(sv::StaticVector{E,T}, x::T2, ::Val{E2}) where {E,T,E2,T2} =
    SVector{E2, T2}(ntuple(i -> i > E ? x : T2(sv[i]), Val(E2)))
padright(sv::StaticVector{E,T}, ::Val{E2}) where {E,T,E2} = padright(sv, zero(T), Val(E2))

@inline padrightbottom(s::SMatrix{E,L}, st::Type{S}) where {E,L,E2,L2,T2,S<:SMatrix{E2,L2,T2}} =
    SMatrix{E2,L2,T2}(
        ntuple(k -> _padrightbottom((k - 1) % E2 + 1, (k - 1) ÷ E2 + 1, zero(T2), s), 
               Val(E2 * L2)))
@inline _padrightbottom(i, j, zero, s::SMatrix{E,L}) where {E,L} = 
    i > E || j > L ? zero : s[i,j]
padrightbottom(m::Matrix{T}, im, jm) where {T} = padrightbottom(m, zero(T), im, jm)
function padrightbottom(m::Matrix{T}, zeroT::T, im, jm) where T
    i0, j0 = size(m)
    [i <= i0 && j<= j0 ? m[i,j] : zeroT for i in 1:im, j in 1:jm]
end

# @inline tuplejoin(x) = x
# @inline tuplejoin(x, y) = (x..., y...)
# @inline tuplejoin(x, y, z...) = (x..., tuplejoin(y, z...)...)
# tuplesort((a,b)::Tuple{<:Number,<:Number}) = a > b ? (b, a) : (a, b)
# tuplesort(t::Tuple) = t
# tuplesort(::Missing) = missing

collectfirst(s::T, ss...) where {T} = _collectfirst((s,), ss...)
_collectfirst(ts::NTuple{N,T}, s::T, ss...) where {N,T} = 
    _collectfirst((ts..., s), ss...)
_collectfirst(ts::Tuple, ss...) = (ts, ss)
_collectfirst(ts::NTuple{N,System}, s::System, ss...) where {N} = 
    _collectfirst((ts..., s), ss...)

## Work around BUG: -SVector{0,Int}() isa SVector{0,Union{}}
negSVector(s::SVector{L,<:Number}) where {L} = -s
negSVector(s::SVector{0,<:Number}) where {L} = s    

allorderedpairs(v) = [(i, j) for i in v, j in v if i >= j]