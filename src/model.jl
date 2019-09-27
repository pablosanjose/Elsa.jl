#######################################################################
# TightbindingModelTerm
#######################################################################
abstract type AbstractTightbindingModel end
abstract type TightbindingModelTerm <: AbstractTightbindingModel end

struct OnsiteTerm{F,
                  S<:Union{Missing,Tuple{Vararg{Int}}},
                  C} <: TightbindingModelTerm
    o::F
    sublats::S
    coefficient::C
    forcehermitian::Bool
end

struct HoppingTerm{F,
                   S<:Union{Missing,Tuple{Vararg{Tuple{Int,Int}}}},
                   D<:Union{Missing,Tuple{Vararg{SVector{L,Int}}} where L},
                   R<:Union{Missing,Real},
                   C} <: TightbindingModelTerm
    h::F
    sublats::S
    dns::D
    range::R
    coefficient::C
    forcehermitian::Bool
end

(o::OnsiteTerm{<:Function})(r) = o.o(r)
(o::OnsiteTerm)(r) = o.o

(h::HoppingTerm{<:Function})(r, dr) = h.h(r, dr)
(h::HoppingTerm)(r, dr) = h.h

normalizesublats(s::Integer) = Tuple(s)
normalizesublats(s::Missing) = missing
normalizesublats(n) = throw(ErrorException(
    "`sublats` for `onsite` must be either `missing`, an `s` or a tuple of `s`s, with `s::Integer` a sublattice number"))

normalizesublatpairs(s::Missing) = missing
normalizesublatpairs((s1, s2)::Tuple{Integer,Integer}) = ((s1, s2),)
normalizesublatpairs((s2, s1)::Pair{<:Integer,<:Integer}) = ((s1, s2),)
normalizesublatpairs(s::Integer) = ((s,s),)
normalizesublatpairs(s::NTuple{N,Any}) where {N} =
    ntuple(n -> first(normalizesublatpairs(s[n])), Val(N))
normalizesublatpairs(s) = throw(ErrorException(
    "`sublats` for `hopping` must be either `missing`, a tuple `(s₁, s₂)`, or a tuple of such tuples, with `sᵢ::Integer` a sublattice number"))

normalizedn(dn::Missing) = missing
normalizedn(dn::Tuple{Vararg{Tuple}}) = SVector.(dn)
normalizedn(dn::Tuple{Vararg{Integer}}) = (SVector(dn),)

sublats(t::OnsiteTerm, lat::AbstractLattice) =
    t.sublats === missing ? collect(1:nsublats(lat)) : t.sublats
sublats(t::HoppingTerm, lat::AbstractLattice) =
    t.sublats === missing ? collect(Iterators.product(1:nsublats(lat), 1:nsublats(lat))) : t.sublats

displayparameter(::Type{<:Function}) = "Function"
displayparameter(::Type{T}) where {T} = "$T"

function Base.show(io::IO, o::OnsiteTerm{F}) where {F}
    i = get(io, :indent, "")
    print(io,
"$(i)OnsiteTerm{$(displayparameter(F))}:
$(i)  Sublattices      : $(o.sublats === missing ? "any" : o.sublats)
$(i)  Force Hermitian  : $(o.forcehermitian)
$(i)  Coefficient      : $(o.coefficient)")
end

function Base.show(io::IO, h::HoppingTerm{F}) where {F}
    i = get(io, :indent, "")
    print(io,
"$(i)HoppingTerm{$(displayparameter(F))}:
$(i)  Sublattice pairs : $(h.sublats === missing ? "any" : (t -> Pair(reverse(t)...)).(h.sublats))
$(i)  dn cell jumps    : $(h.dns === missing ? "any" : h.dns)
$(i)  Hopping range    : $(round(h.range, digits = 6))
$(i)  Force Hermitian  : $(h.forcehermitian)
$(i)  Coefficient      : $(h.coefficient)")
end

# External API #

function onsite(o; sublats = missing, forcehermitian::Bool = true)
    return OnsiteTerm(o, normalizesublats(sublats), 1, forcehermitian)
end

function hopping(h; sublats = missing, range::Real = 1, dn = missing, forcehermitian::Bool = true)
    return HoppingTerm(h, normalizesublatpairs(sublats), normalizedn(dn),
                       range + sqrt(eps(Float64)), 1, forcehermitian)
end

Base.:*(x, o::OnsiteTerm) =
    OnsiteTerm(o.o, o.sublats, x * o.coefficient, o.forcehermitian)
Base.:*(x, t::HoppingTerm) =
    HoppingTerm(t.h, t.sublats, t.dns, t.range, x * t.coefficient, t.forcehermitian)
Base.:*(t::TightbindingModelTerm, x) = x * t
Base.:-(t::TightbindingModelTerm) = (-1) * t

Base.:+(t1::TightbindingModelTerm, t2::TightbindingModelTerm) = TightbindingModel((t1, t2))
Base.:-(t1::TightbindingModelTerm, t2::TightbindingModelTerm) = TightbindingModel((t1, -t2))


#######################################################################
# TightbindingModel
#######################################################################
struct TightbindingModel{N,T<:Tuple{Vararg{TightbindingModelTerm,N}}} <: AbstractTightbindingModel
    terms::T
end

terms(t::TightbindingModel) = t.terms
terms(t::TightbindingModelTerm) = (t,)

TightbindingModel(t::AbstractTightbindingModel...) = TightbindingModel(tuplejoin(terms.(t)...))

# External API #

Base.:*(x, m::TightbindingModel) = TightbindingModel(x .* m.terms)
Base.:*(m::TightbindingModel, x) = x * m
Base.:-(m::TightbindingModel) = TightbindingModel((-1) .* m.terms)

Base.:+(m::TightbindingModel, t::TightbindingModelTerm) = TightbindingModel((m.terms..., t))
Base.:+(t::TightbindingModelTerm, m::TightbindingModel) = TightbindingModel((t, m.terms...))
Base.:-(m::TightbindingModel, t::TightbindingModelTerm) = m + (-t)
Base.:-(t::TightbindingModelTerm, m::TightbindingModel) = t + (-m)

function Base.show(io::IO, m::TightbindingModel{N,F}) where {N,F}
    ioindent = IOContext(io, :indent => "  ")
    print(io, "TightbindingModel{$N,$F}: $N terms, of which $F can be fused:", "\n")
    foreach(t -> print(ioindent, t, "\n"), m.terms)
end