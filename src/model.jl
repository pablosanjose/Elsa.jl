#######################################################################
# ModelBlock
#######################################################################
abstract type ModelTerm{F,S} end
struct Onsite{F,S} <: ModelTerm{F,S}
    o::F
    sublats::S              # SL === Missing means any sublats 
end
struct Hopping{F,S,D,R} <: ModelTerm{F,S}
    h::F
    sublats::S              # S === Missing means any sublats 
    ndists::D               # D  === Missing means any ndist
    range::R 
end

isonsite(term::Onsite) = true
isonsite(term::Hopping) = false

(o::Onsite{F})(r, dr) where {F<:Function} = ensureSMatrix((o.o(r) + o.o(r)')/2)
(o::Onsite{S})(r, dr) where {S} = ensureSMatrix((o.o + o.o')/2)
(h::Hopping{F})(r, dr) where {F<:Function} = ensureSMatrix(h.h(r, dr))
(h::Hopping{S})(r, dr) where {S}  = ensureSMatrix(h.h)

function Base.show(io::IO, o::Onsite)
    print(io, "ModelTerm: onsite term on sublats = $(o.sublats)")
end

function Base.show(io::IO, h::Hopping)
    print(io, "ModelTerm: hopping term between sublat pairs = $(h.sublats), Bravais ndists = $(h.ndists) and maximum range = $(round(h.range, digits = 6))")
end

#######################################################################
# Model
#######################################################################
"""
    Model(terms::ModelTerm...)
    Model{Tv}(terms::ModelTerm...)

Define a `Model` of numeric type `Tv` (`Complex{Float64}` by default) from a collection of 
`terms`, of type `Onsite` or `Hopping`. See the latter for more information.

# See also
    `Onsite`, `Hopping`
"""
struct Model{Tv,N,MB<:NTuple{N,ModelTerm}}
    terms::MB
end
Model{Tv}(hs::Vararg{ModelTerm,N}) where {Tv,N} = Model{Tv,N,typeof(hs)}(hs)
Model(hs...) = Model{Complex{Float64}}(hs...)
nonsites(m::Model) = count(isonsite, m.terms)
nhoppings(m::Model) = count(!isonsite, m.terms)

function Base.show(io::IO, m::Model{Tv,N}) where {Tv,N}
    print(io, "Model{$Tv,$N}: Model with $N terms ($(nonsites(m)) onsite and $(nhoppings(m)) hopping)")
end

"""
    Onsite(o; sublats = missing)

Define a `Onsite<:ModelTerm` that models an onsite energy `o` at sites on specific 
`sublats`. Wrap several `ModelBlock`s in a `Model` to apply to a system.
 
 `o` can be a `o::SMatrix{N,N,Tv}` of dimension `NxN`, a number  `o::Tv`, or a function of 
 position (`r -> o(r)`). The dimensions `N` of `o` will be checked for consistency between
 different terms when building a `System` with the model.
 
 `sublats` can be a single or a tuple of sublattice numbers or sublattice names. If 
 `missing`, all sublattices are included.
"""
Onsite(o; sublats = missing) = _onsite(o, _normaliseSL(sublats))
_onsite(o, sublats) = Onsite(o, sublats)
_onsite(o::SMatrix{N,N}, sublats) where {N} = Onsite(o, sublats)
_onsite(o::SMatrix{N,M}, sublats) where {N,M} = throw(
    DimensionMismatch("Onsite energy must be a scalar or a square matrix"))
_onsite(o::AbstractArray, sublats) = Onsite(ensureSMatrix(o), sublats)

"""
    Hopping(h; sublats = missing, ndists = missing, range = 1)

Define a `Hopping<:ModelTerm` that models a hopping `t` between sites on specific `sublats` 
and at a specific `range` and/or `ndists`. Wrap several `ModelTerms`s in a `Model` to apply 
to a system.

`h` can be a `h::SMatrix{N,M,Tv}` of dimension `NxM`, a number `h::Tv`, or a function of two 
positions (`(r,dr) -> h(r)`). Here, `r = (r1 + r2)/2` is the center of a link between two 
sites (at `r1`, `r2`), while `dr = r2 - r1` is the link vector. The dimensions `N,M` of `h` 
will be checked for consistency between different terms when building a `System` with the 
model.

`sublats` can be a single or pair of sublattice numbers, sublattice 
name or a tuple of pairs. If `missing`, all sublattices are included. 

`ndists::Union{NTuple{D,NTuple{L,Int}},Missing}` specifies a tuple of 
`D` unit cell distances (each a tuple of `L` integers) at which to 
apply the hopping. If `missing` all `ndist`s will be included.

`range::Union{Real,Missing}` is a maximum Euclidean distance of sites to 
link, or any distance if `missing`.
"""
Hopping(h; sublats = missing, ndists = missing, range = 1) =
    Hopping(ensureSMatrix(h), _normaliseSLpairs(sublats), _normaliseND(ndists), 
    range + extended_eps(Float64))

_normaliseND(::Missing) = missing
_normaliseND(n::NTuple{L,Int}) where L = [SVector{L,Int}(n)]
_normaliseND(n::NTuple{N,NTuple{L,Int}}) where {N,L} = [SVector{L,Int}.(n)...]
_normaliseND(n::Vector{NTuple{L,Int}}) where {L} = SVector{L,Int}.(n)
_normaliseND(n) = throw(ErrorException(
    "`ndists` must be either `missing`, a `NTuple{L,Int}` or a list of them"))

_normaliseSLpairs(::Missing) = missing
_normaliseSLpairs(s::Union{Int,NameType}) = ((s,s),)
_normaliseSLpairs(ss::NTuple{N, Any}) where N = 
    ntuple(n -> first(_normaliseSLpairs(ss[n])), Val(N))
_normaliseSLpairs(ss::Tuple{Union{Int,NameType}, Union{Int,NameType}}) = (ss,)
_normaliseSLpairs(n) = throw(ErrorException(
    "`sublats` must be either `missing` or a tuple of `s::Union{S, NTuple{2,S}}`, where `S<:Union{Int,$(NameType)}` specifies a sublattice number or name"))

_normaliseSL(::Missing) = missing
_normaliseSL(s::Union{Int,NameType}) = ((s,s),)
_normaliseSL(ss::NTuple{N, Any}) where N = ntuple(n -> first(_normaliseSL(ss[n])), Val(N))
_normaliseSL(n) = throw(ErrorException("`sublats` must be either `missing` or a tuple of `s::Union{S, NTuple{2,S}}`, where `S<:Union{Int,$(NameType)}` specifies a sublattice number or name"))