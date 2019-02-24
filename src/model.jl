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
# struct ConjHopping{F,H<:Hopping{F}} <:ModelTerm{F}
#     h::H
# end

isonsite(term::Onsite) = true
isonsite(term::Hopping) = false
# isonsite(term::Union{Hopping,ConjHopping}) = false

(o::Onsite{F})(r, dr) where {F<:Function} = ensureSMatrix((o.o(r) + o.o(r)')/2)
(o::Onsite{S})(r, dr) where {S} = ensureSMatrix((o.o + o.o')/2)
(h::Hopping{F})(r, dr) where {F<:Function} = ensureSMatrix(h.h(r, dr))
(h::Hopping{S})(r, dr) where {S}  = ensureSMatrix(h.h)
# (h::ConjHopping{F})(r, dr) where {F<:Function} = ensureSMatrix(h.h.h(r, dr))'
# (h::ConjHopping{S})(r, dr) where {S}  = ensureSMatrix(h.h.h)'

# Base.adjoint(h::Hopping) = ConjHopping(h)
# Base.adjoint(h::ConjHopping) = h.h

function Base.show(io::IO, o::Onsite)
    print(io, "Onsite model term on sublats = $(o.sublats)")
end

function Base.show(io::IO, h::Hopping)
    print(io, "Hopping model term between sublat pairs = $(h.sublats), Bravais ndists = $(h.ndists) and maximum range = $(round(h.range, digits = 6))")
end

#######################################################################
# Model
#######################################################################
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
    onsite(v; sublats = missing)

Create a `ModelBlock` that models an onsite energy `v` at sites on 
specific `sublats`. Wrap several `ModelBlock`s in a `Model` to apply to a 
system.
 
 `v` can be a `t::AbstractMatrix{Tv}` of dimension `NxN`, a number 
 `v::Tv`, or a function of position (`r -> v(r)`). The dimensions 
 `N` of `v` must match the number of orbitals declared for all 
 specified sublattices. 
 
 `sublats` can be a single or a tuple of sublattice numbers or 
sublattice names. If `missing`, all sublattices are included.
"""
onsite(o; sublats = missing) = Onsite(o, _normaliseSL(sublats))

"""
    hopping(t; sublats = missing, ndists = missing, range = 1)

Create a `ModelBlock` that models a hopping `t` between sites 
on specific `sublats` and at a specific `range` and/or `ndists`. 
Wrap several `ModelBlock`s in a `Model` to apply to a system.

`t` can be a `t::AbstractMatrix{Tv}` of dimension `NxM`, a number 
`t::Tv`, or a function of position (`r -> hopping(r)`). The dimensions 
`NxM` of `hopping` must match the number of orbitals declared for all 
specified sublattices.

`sublats` can be a single or pair of sublattice numbers, sublattice 
name or a tuple of pairs. If `missing`, all sublattices are included. 

`ndists::Union{NTuple{D,NTuple{L,Int}},Missing}` specifies a tuple of 
`D` unit cell distances (each a tuple of `L` integers) at which to 
apply the hopping. If `missing` all `ndist`s will be included.

`range::Union{Real,Missing}` is a maximum Euclidean distance of sites to 
link, or any distance if `missing`.
"""
hopping(h; sublats = missing, ndists = missing, range = 1) =
    Hopping(h, _normaliseSLpairs(sublats), _normaliseND(ndists), 
    range + extended_eps(Float64))


_normaliseND(::Missing) = missing
_normaliseND(n::NTuple{L,Int}) where L = [SVector{L,Int}(n)]
_normaliseND(n::NTuple{N,NTuple{L,Int}}) where {N,L} = [SVector{L,Int}.(n)...]
_normaliseND(n::Vector{NTuple{L,Int}}) where {L} = SVector{L,Int}.(n)
_normaliseND(n) = throw(ErrorException("`ndists` must be either `missing`, a `NTuple{L,Int}` or a list of them"))

_normaliseSLpairs(::Missing) = missing
_normaliseSLpairs(s::Union{Int,NameType}) = ((s,s),)
_normaliseSLpairs(ss::NTuple{N, Any}) where N = ntuple(n -> first(_normaliseSLpairs(ss[n])), Val(N))
_normaliseSLpairs(ss::Tuple{Union{Int,NameType}, Union{Int,NameType}}) = (ss,)
_normaliseSLpairs(n) = throw(ErrorException("`sublats` must be either `missing` or a tuple of `s::Union{S, NTuple{2,S}}`, where `S<:Union{Int,NameType}` specifies a sublattice number or name"))

_normaliseSL(::Missing) = missing
_normaliseSL(s::Union{Int,NameType}) = ((s,s),)
_normaliseSL(ss::NTuple{N, Any}) where N = ntuple(n -> first(_normaliseSL(ss[n])), Val(N))
_normaliseSL(n) = throw(ErrorException("`sublats` must be either `missing` or a tuple of `s::Union{S, NTuple{2,S}}`, where `S<:Union{Int,$(NameType)}` specifies a sublattice number or name"))