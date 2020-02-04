#######################################################################
# Onsite/Hopping selectors
#######################################################################
abstract type Selector{M,S} end

struct OnsiteSelector{M,S} <: Selector{M,S}
    region::M
    sublats::S  # NTuple{N,NameType} (unresolved) or Vector{Int} (resolved on a lattice)
end

struct HoppingSelector{M,S,D,T} <: Selector{M,S}
    region::M
    sublats::S  # NTuple{N,Tuple{NameType,NameType}} (unres) or Vector{Tuple{Int,Int}} (res)
    dns::D
    range::T
end

"""
    onsiteselector(; region = missing, sublats = missing)

Specifies a subset of onsites energies in a given hamiltonian. Only sites at position `r` in
sublattice with name `s::NameType` will be selected if `region(r) && s in sublats` is true.
Any missing `region` or `sublat` will not be used to constraint the selection.

# See also:
    `hoppingselector`, `onsite`, `hopping`
"""
onsiteselector(; region = missing, sublats = missing) =
    OnsiteSelector(region, sanitize_sublats(sublats))

"""
    hoppingselector(; region = missing, sublats = missing, dn = missing, range = missing)

Specifies a subset of hoppings in a given hamiltonian. Only hoppings between two sites at
positions `r₁ = r - dr/2` and `r₂ = r + dr`, belonging to unit cells at integer
distance `dn´` and to sublattices `s₁` and `s₂` will be selected if: `region(r, dr) && s in
sublats && dn´ in dn && norm(dr) <= range`. If any of these is `missing` it will not be used
to constraint the selection.

# See also:
    `onsiteselector`, `onsite`, `hopping`
"""
hoppingselector(; region = missing, sublats = missing, dn = missing, range = missing) =
    HoppingSelector(region, sanitize_sublatpairs(sublats), sanitize_dn(dn), sanitize_range(range))

sanitize_sublats(s::Missing) = missing
sanitize_sublats(s::Integer) = (nametype(s),)
sanitize_sublats(s::NameType) = (s,)
sanitize_sublats(s::Tuple) where {N} = nametype.(s)
sanitize_sublats(s::Tuple{}) = ()
sanitize_sublats(n) = throw(ErrorException(
    "`sublats` for `onsite` must be either `missing`, an `s` or a tuple of `s`s, with `s::$NameType` is a sublattice name"))

sanitize_sublatpairs(s::Missing) = missing
sanitize_sublatpairs((s1, s2)::NTuple{2,Union{Integer,NameType}}) = ((nametype(s1), nametype(s2)),)
sanitize_sublatpairs((s2, s1)::Pair) = sanitize_sublatpairs((s1, s2))
sanitize_sublatpairs(s::Union{Integer,NameType}) = sanitize_sublatpairs((s,s))
sanitize_sublatpairs(s::NTuple{N,Any}) where {N} =
    ntuple(n -> first(sanitize_sublatpairs(s[n])), Val(N))
sanitize_sublatpairs(s) = throw(ErrorException(
    "`sublats` for `hopping` must be either `missing`, a tuple `(s₁, s₂)`, or a tuple of such tuples, with `sᵢ::$NameType` a sublattice name"))

sanitize_dn(dn::Missing) = missing
sanitize_dn(dn::Tuple{Vararg{Tuple}}) = SVector.(dn)
sanitize_dn(dn::Tuple{Vararg{Integer}}) = (SVector(dn),)
sanitize_dn(dn::Tuple{}) = ()

sanitize_range(::Missing) = missing
sanitize_range(range::Real) = float(range) + sqrt(eps(float(range)))

sublats(s::OnsiteSelector{<:Any,Missing}, lat::AbstractLattice) = collect(1:nsublats(lat))

function sublats(s::OnsiteSelector{<:Any,<:Tuple}, lat::AbstractLattice)
    names = lat.unitcell.names
    ss = Int[]
    for name in s.sublats
        i = findfirst(isequal(name), names)
        i !== nothing && push!(ss, i)
    end
    return ss
end

sublats(s::HoppingSelector{<:Any,Missing}, lat::AbstractLattice) =
    vec(collect(Iterators.product(1:nsublats(lat), 1:nsublats(lat))))

function sublats(s::HoppingSelector{<:Any,<:Tuple}, lat::AbstractLattice)
    names = lat.unitcell.names
    ss = Tuple{Int,Int}[]
    for (n1, n2) in s.sublats
        i1 = findfirst(isequal(n1), names)
        i2 = findfirst(isequal(n2), names)
        i1 !== nothing && i2 !== nothing && push!(ss, (i1, i2))
    end
    return ss
end

# selector already resolved for a lattice
sublats(s::Selector{<:Any,<:Vector}, lat) = s.sublats

# API

resolve(s::HoppingSelector, lat::Lattice) =
    HoppingSelector(s.region, sublats(s, lat), _checkdims(s.dns, lat), s.range)
resolve(s::OnsiteSelector, lat::Lattice) = OnsiteSelector(s.region, sublats(s, lat))

_checkdims(dns::Missing, lat::Lattice{E,L}) where {E,L} = dns
_checkdims(dns::Tuple{Vararg{SVector{L,Int}}}, lat::Lattice{E,L}) where {E,L} = dns
_checkdims(dns, lat::Lattice{E,L}) where {E,L} =
    throw(DimensionMismatch("Specified cell distance `dn` does not match lattice dimension $L"))

(s::OnsiteSelector)(lat::Lattice, (i, j)::Tuple, dn::SVector) =
    i == j && iszero(dn) && isinregion(i, s.region, lat) && isinsublats(sublat(lat, i), s.sublats)

(s::HoppingSelector)(lat::Lattice, inds, dn) =
    isinregion(inds, s.region, lat) && isindns(dn, s.dns) &&
    isinrange(inds, s.range, lat) && isinsublats(sublat.(Ref(lat), inds), s.sublats)

isinregion(i::Int, ::Missing, lat) = true
isinregion(i::Int, region::Function, lat) = region(sites(lat)[i])
isinregion(is::Tuple{Int,Int}, ::Missing, lat) = true
function isinregion((src, dst)::Tuple{Int,Int}, region::Function, lat)
    r, dr = _rdr(sites(lat)[src], sites(lat)[dst])
    return region(r, dr)
end

isinsublats(s::Int, ::Missing) = true
isinsublats(s::Int, sublats::Vector{Int}) = s in sublats
isinsublats(ss::Tuple{Int,Int}, ::Missing) = true
isinsublats(ss::Tuple{Int,Int}, sublats::Vector{Tuple{Int,Int}}) = ss in sublats
isinsublats(s, sublats) =
    throw(ArgumentError("Sublattices $sublats in selector are not resolved."))

isindns(dn::SVector{L,Int}, dns::Tuple{Vararg{SVector{L,Int}}}) where {L} = dn in dns
isindns(dn, dns) =
    throw(ArgumentError("Cell distance dn in selector is incompatible with Lattice."))

isinrange(inds, ::Missing, lat) = true
isinrange((src, dst)::Tuple{Int,Int}, range, lat) =
    norm(sites(lat)[dst] - sites(lat)[src]) <= range

#######################################################################
# TightbindingModelTerm
#######################################################################
abstract type TightbindingModelTerm end
abstract type AbstractOnsiteTerm <: TightbindingModelTerm end
abstract type AbstractHoppingTerm <: TightbindingModelTerm end

struct OnsiteTerm{F,S<:OnsiteSelector,C} <: AbstractOnsiteTerm
    o::F
    selector::S
    coefficient::C
    forcehermitian::Bool
end

struct HoppingTerm{F,S<:HoppingSelector,C} <: AbstractHoppingTerm
    t::F
    selector::S
    coefficient::C
    forcehermitian::Bool
end

(o::OnsiteTerm{<:Function})(r,dr) = o.coefficient * o.o(r)
(o::OnsiteTerm)(r,dr) = o.coefficient * o.o

(h::HoppingTerm{<:Function})(r, dr) = h.coefficient * h.t(r, dr)
(h::HoppingTerm)(r, dr) = h.coefficient * h.t

sublats(t::TightbindingModelTerm, lat) = sublats(t.selector, lat)

displayparameter(::Type{<:Function}) = "Function"
displayparameter(::Type{T}) where {T} = "$T"

function Base.show(io::IO, o::OnsiteTerm{F}) where {F}
    i = get(io, :indent, "")
    print(io,
"$(i)OnsiteTerm{$(displayparameter(F))}:
$(i)  Sublattices      : $(o.selector.sublats === missing ? "any" : o.selector.sublats)
$(i)  Force hermitian  : $(o.forcehermitian)
$(i)  Coefficient      : $(o.coefficient)")
end

function Base.show(io::IO, h::HoppingTerm{F}) where {F}
    i = get(io, :indent, "")
    print(io,
"$(i)HoppingTerm{$(displayparameter(F))}:
$(i)  Sublattice pairs : $(h.selector.sublats === missing ? "any" : (t -> Pair(reverse(t)...)).(h.selector.sublats))
$(i)  dn cell distance : $(h.selector.dns === missing ? "any" : h.selector.dns)
$(i)  Hopping range    : $(round(h.selector.range, digits = 6))
$(i)  Force hermitian  : $(h.forcehermitian)
$(i)  Coefficient      : $(h.coefficient)")
end

# External API #
"""
    onsite(o; forcehermitian = true, kw...)
    onsite(o, onsiteselector(; kw...); forcehermitian = true)

Create an `TightbindingModelTerm` that applies an onsite energy `o` to a `Lattice` when
creating a `Hamiltonian` with `hamiltonian`. A subset of sites can be specified with the
`kw...`, see `onsiteselector` for details.

The onsite energy `o` can be a number, a matrix (preferably `SMatrix`) or a function of the
form `r -> ...` for a position-dependent onsite energy. If `forcehermitian` is true, the
model will produce an hermitian Hamiltonian.

The dimension of `o::AbstractMatrix` must match the orbital dimension of applicable
sublattices (see also `orbitals` option for `hamiltonian`). If `o::Number` it will be
treated as `o * I` (proportional to identity matrix) when applied to multiorbital
sublattices.

`TightbindingModelTerm`s created with `onsite` or `hopping` can be added or substracted
together to build more complicated `TightbindingModel`s.

# Examples
```
julia> onsite(1, sublats = (:A,:B)) - hopping(2, sublats = :A=>:A)
TightbindingModel{2}: model with 2 terms
  OnsiteTerm{Int64}:
    Sublattices      : (:A, :B)
    Force hermitian  : true
    Coefficient      : 1
  HoppingTerm{Int64}:
    Sublattice pairs : (:A => :A,)
    dn cell distance : any
    Hopping range    : 1.0
    Force hermitian  : true
    Coefficient      : -1

julia> LatticePresets.honeycomb() |> hamiltonian(onsite(r->@SMatrix[1 2; 3 4]), orbitals = Val(2))
Hamiltonian{<:Lattice} : Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 1 (SparseMatrixCSC, sparse)
  Harmonic size    : 2 × 2
  Orbitals         : ((:a, :a), (:a, :a))
  Element type     : 2 × 2 blocks (Complex{Float64})
  Onsites          : 2
  Hoppings         : 0
  Coordination     : 0.0
```

# See also:
    `hopping`, `onsiteselector`, `hoppingselector`
"""
onsite(o; forcehermitian = true, kw...) =
    onsite(o, onsiteselector(; kw...); forcehermitian = forcehermitian)
onsite(o, selector; forcehermitian::Bool = true) =
    TightbindingModel(OnsiteTerm(o, selector, 1, forcehermitian))

"""
    hopping(t; forcehermitian = true, range = 1, kw...)
    hopping(t, hoppingselector(; range = 1, kw...); forcehermitian = true)

Create an `TightbindingModelTerm` that applies a hopping `t` to a `Lattice` when
creating a `Hamiltonian` with `hamiltonian`. A subset of hoppings can be specified with the
`kw...`, see `hoppingselector` for details. Note that a default `range = 1` is assumed.

The hopping amplitude `t` can be a number, a matrix (preferably `SMatrix`) or a function
of the form `(r, dr) -> ...` for a position-dependent hopping (`r` is the bond center,
and `dr` the bond vector). If `sublats` is specified as a sublattice name pair, or tuple
thereof, `hopping` is only applied between sublattices with said names. If `forcehermitian`
is true, the model will produce an hermitian Hamiltonian.

The dimension of `t::AbstractMatrix` must match the orbital dimension of applicable
sublattices (see also `orbitals` option for `hamiltonian`). If `t::Number` it will be
treated as `t * I` (proportional to identity matrix) when applied to multiorbital
sublattices.

`TightbindingModelTerm`s created with `onsite` or `hopping` can be added or substracted
together to build more complicated `TightbindingModel`s.

# Examples
```
julia> onsite(1) - hopping(2, dn = ((1,2), (0,0)), sublats = :A=>:B)
TightbindingModel{2}: model with 2 terms
  OnsiteTerm{Int64}:
    Sublattices      : any
    Force hermitian  : true
    Coefficient      : 1
  HoppingTerm{Int64}:
    Sublattice pairs : (:A => :B,)
    dn cell distance : ([1, 2], [0, 0])
    Hopping range    : 1.0
    Force hermitian  : true
    Coefficient      : -1

julia> LatticePresets.honeycomb() |> hamiltonian(hopping((r,dr) -> cos(r[1]), sublats = ((:A,:A), (:B,:B))))
Hamiltonian{<:Lattice} : Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 7 (SparseMatrixCSC, sparse)
  Harmonic size    : 2 × 2
  Orbitals         : ((:a,), (:a,))
  Element type     : scalar (Complex{Float64})
  Onsites          : 0
  Hoppings         : 12
  Coordination     : 6.0
```

# See also:
    `onsite`, `onsiteselector`, `hoppingselector`
"""
hopping(t; forcehermitian = true, range = 1, kw...) =
    hopping(t, hoppingselector(; range = range, kw...); forcehermitian = forcehermitian)
hopping(t, selector; forcehermitian = true) =
    TightbindingModel(HoppingTerm(t, selector, 1, forcehermitian))

Base.:*(x, o::OnsiteTerm) =
    OnsiteTerm(o.o, o.selector, x * o.coefficient, o.forcehermitian)
Base.:*(x, t::HoppingTerm) =
    HoppingTerm(t.t, t.selector, x * t.coefficient, t.forcehermitian)
Base.:*(t::TightbindingModelTerm, x) = x * t
Base.:-(t::TightbindingModelTerm) = (-1) * t

LinearAlgebra.ishermitian(t::OnsiteTerm) = t.forcehermitian
LinearAlgebra.ishermitian(t::HoppingTerm) = t.forcehermitian

#######################################################################
# TightbindingModel
#######################################################################
struct TightbindingModel{N,T<:Tuple{Vararg{TightbindingModelTerm,N}}}
    terms::T
end

terms(t::TightbindingModel) = t.terms

TightbindingModel(ts::TightbindingModelTerm...) = TightbindingModel(ts)

(m::TightbindingModel)(r, dr) = sum(t -> t(r, dr), m.terms)

# External API #

Base.:*(x, m::TightbindingModel) = TightbindingModel(x .* m.terms)
Base.:*(m::TightbindingModel, x) = x * m
Base.:-(m::TightbindingModel) = TightbindingModel((-1) .* m.terms)

Base.:+(m::TightbindingModel, t::TightbindingModel) = TightbindingModel((m.terms..., t.terms...))
Base.:-(m::TightbindingModel, t::TightbindingModel) = m + (-t)

function Base.show(io::IO, m::TightbindingModel{N}) where {N}
    ioindent = IOContext(io, :indent => "  ")
    print(io, "TightbindingModel{$N}: model with $N terms", "\n")
    foreach(t -> print(ioindent, t, "\n"), m.terms)
end

LinearAlgebra.ishermitian(m::TightbindingModel) = all(t -> ishermitian(t), m.terms)

#######################################################################
# offdiagonal
#######################################################################
"""
    offdiagonal(model, lat, nsublats::NTuple{N,Int})

Build a restricted version of `model` that applies only to off-diagonal blocks formed by
sublattice groups of size `nsublats`.
"""
offdiagonal(m::TightbindingModel, lat, nsublats) =
    TightbindingModel(offdiagonal.(m.terms, Ref(lat), Ref(nsublats)))

offdiagonal(o::OnsiteTerm, lat, nsublats) =
    throw(ArgumentError("No onsite terms allowed in off-diagonal coupling"))

function offdiagonal(t::HoppingTerm, lat, nsublats)
    selector´ = resolve(t.selector, lat)
    s = selector´.sublats
    sr = sublatranges(nsublats...)
    filter!(spair ->  findblock(first(spair), sr) != findblock(last(spair), sr), s)
    return HoppingTerm(t.t, selector´, t.coefficient, t.forcehermitian)
end

sublatranges(i::Int, is::Int...) = _sublatranges((1:i,), is...)
_sublatranges(rs::Tuple, i::Int, is...) = _sublatranges((rs..., last(last(rs)) + 1: last(last(rs)) + i), is...)
_sublatranges(rs::Tuple) = rs

findblock(s, sr) = findfirst(r -> s in r, sr)

#######################################################################
# checkmodelorbs - check for inconsistent orbital dimensions
#######################################################################
checkmodelorbs(model::TightbindingModel, orbs, lat) =
    foreach(term -> _checkmodelorbs(term, orbs, lat), model.terms)

function _checkmodelorbs(term::HoppingTerm, orbs, lat)
    for (s1, s2) in sublats(term, lat)
        _checkmodelorbs(term(first(sites(lat, s1)), first(sites(lat, s2))), length(orbs[s1]), length(orbs[s2]))
    end
    return nothing
end

function _checkmodelorbs(term::OnsiteTerm, orbs, lat)
    for s in sublats(term, lat)
        _checkmodelorbs(term(first(sites(lat, s)), first(sites(lat, s))), length(orbs[s]))
    end
    return nothing
end

_checkmodelorbs(s::SMatrix, m, n = m) =
    size(s) == (m, n) || @warn("Possible dimension mismatch between model and Hamiltonian. Did you correctly specify the `orbitals` in hamiltonian?")

_checkmodelorbs(s::Number, m, n = m) = _checkmodelorbs(SMatrix{1,1}(s), m, n)

#######################################################################
# onsite! and hopping!
#######################################################################
abstract type ElementModifier end

struct Onsite!{V<:Val,F<:Function,S<:Selector} <: ElementModifier
    f::F
    needspositions::V    # Val{false} for f(o; kw...), Val{true} for f(o, r; kw...) or other
    selector::S
end

Onsite!(f, selector) = Onsite!(f, Val(!applicable(f, 0.0)), selector)

struct Hopping!{V<:Val,F<:Function,S<:Selector} <: ElementModifier
    f::F
    needspositions::V    # Val{false} for f(h; kw...), Val{true} for f(h, r, dr; kw...) or other
    selector::S
end

Hopping!(f, selector) = Hopping!(f, Val(!applicable(f, 0.0)), selector)

# API #

"""
    onsite!(f; kw...)
    onsite!(f, onsiteselector(; kw...))

Create an `ElementModifier`, to be used with `parametric`, that applies `f` to onsite
energies specified by `onsiteselector(; kw...)`. The form of `f` may be `f = (o; kw...) ->
...` or `f = (o, r; kw...) -> ...` if the modification is position (`r`) dependent. The
former is naturally more efficient, as there is no need to compute the positions of each
onsite energy.

# See also:
    `hopping!`, `parametric`
"""
onsite!(f; kw...) = onsite!(f, onsiteselector(; kw...))
onsite!(f, selector) = Onsite!(f, selector)

"""
    hopping!(f; kw...)
    hopping!(f, hoppingselector(; kw...))

Create an `ElementModifier`, to be used with `parametric`, that applies `f` to hoppings
specified by `hoppingselector(; kw...)`. The form of `f` may be `f = (t; kw...) ->
...` or `f = (t, r, dr; kw...) -> ...` if the modification is position (`r, dr`) dependent. The
former is naturally more efficient, as there is no need to compute the positions of the
two sites involved in each hopping.

# See also:
    `onsite!`, `parametric`
"""
hopping!(f; kw...) = onsite!(f, onsiteselector(; kw...))
hopping!(f, selector) = Hopping!(f, selector)