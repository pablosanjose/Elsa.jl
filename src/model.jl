#######################################################################
# Onsite/Hopping selectors
#######################################################################
abstract type Selector{M,S} end

struct SelectOnsites{M,S} <: Selector{M,S}
    mask::M
    sublats::S
end

struct SelectHoppings{M,S,D,T} <: Selector{M,S}
    mask::M
    sublats::S
    dns::D
    range::T
end

selectonsites(; mask = missing, sublats = missing) =
    SelectOnsites(mask, sanitize_sublats(sublats))

selecthoppings(; mask = missing, sublats = missing, dns = missing, range = missing) =
    SelectHoppings(mask, sanitize_sublatpairs(sublats), sanitize_dn(missing), sanitize_range(range))

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

sublats(s::SelectOnsites{<:Any,Missing}, lat::AbstractLattice) = collect(1:nsublats(lat))

function sublats(s::SelectOnsites{<:Any,<:Tuple}, lat::AbstractLattice)
    names = lat.unitcell.names
    ss = Int[]
    for name in s.sublats
        i = findfirst(isequal(name), names)
        i !== nothing && push!(ss, i)
    end
    return ss
end

sublats(s::SelectHoppings{<:Any,Missing}, lat::AbstractLattice) =
    vec(collect(Iterators.product(1:nsublats(lat), 1:nsublats(lat))))

function sublats(s::SelectHoppings{<:Any,<:Tuple}, lat::AbstractLattice)
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

resolve(s::SelectHoppings, lat::Lattice) =
    SelectHoppings(s.mask, sublats(s, lat), _checkdims(s.dns, lat), s.range)
resolve(s::SelectOnsites, lat::Lattice) = SelectOnsites(s.mask, sublats(s, lat))

_checkdims(dns::Tuple{Vararg{SVector{L,Int}}}, lat::Lattice{E,L}) where {E,L} = dns
_checkdims(dns, lat::Lattice{E,L}) where {E,L} =
    throw(DimensionMismatch("Specified cell distance `dns` does not match lattice dimension $L"))

(s::SelectOnsites)(ind, sublat, lat::Lattice) =
    isinmask(ind, s.mask, lat) && isinsublats(sublat, s.sublats)

(s::SelectHoppings)(inds, sublats, dn, lat::Lattice) =
    isinmask(inds, s.mask, lat) && isinsublats(sublats, s.sublats) &&
    isindns(dn, s.dns) && isinrange(inds, s.range, lat)

isinmask(i::Int, ::Missing, lat) = true
isinmask(i::Int, mask::Function, lat) = mask(sites(lat)[i])
isinmask(is::Tuple{Int,Int}, ::Missing, lat) = true
function isinmask((src, dst)::Tuple{Int,Int}, mask::Function, lat)
    r, dr = _rdr(sites(lat)[src], sites(lat)[dst])
    return mask(r, dr)
end

isinsublats(s::Int, ::Missing) = true
isinsublats(s::Int, sublats::Tuple{Vararg{Int}}) = s in sublats
isinsublats(ss::Tuple{Int,Int}, ::Missing) = true
isinsublats(ss::Tuple{Int,Int}, sublats::Tuple{Vararg{Tuple{Int,Int}}}) = ss in sublats
isinsublats(s, sublats) =
    throw(ArgumentError("Sublattices in selector are not resolved, do `resolve(selector, lattice)` first."))

isindns(dn::SVector{L,Int}, dns::Tuple{Vararg{SVector{L,Int}}}) where {L} = dn in dns
isindns(dn, dns) =
    throw(ArgumentError("Cell distances dns in selector are incompatible with Lattice, do `resolve(selector, lattice)` first."))

isinrange(inds, ::Missing, lat) = true
isinrange((src, dst)::Tuple{Int,Int}, range, lat) =
    norm(sites(lat)[dst] - sites(lat)[src]) <= range

#######################################################################
# TightbindingModelTerm
#######################################################################
abstract type TightbindingModelTerm end
abstract type AbstractOnsiteTerm <: TightbindingModelTerm end
abstract type AbstractHoppingTerm <: TightbindingModelTerm end

struct OnsiteTerm{F,S<:SelectOnsites,C} <: AbstractOnsiteTerm
    o::F
    selector::S
    coefficient::C
    forcehermitian::Bool
end

struct HoppingTerm{F,S<:SelectHoppings,C} <: AbstractHoppingTerm
    h::F
    selector::S
    coefficient::C
    forcehermitian::Bool
end

(o::OnsiteTerm{<:Function})(r,dr) = o.coefficient * o.o(r)
(o::OnsiteTerm)(r,dr) = o.coefficient * o.o

(h::HoppingTerm{<:Function})(r, dr) = h.coefficient * h.h(r, dr)
(h::HoppingTerm)(r, dr) = h.coefficient * h.h

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
$(i)  dn cell jumps    : $(h.selector.dns === missing ? "any" : h.selector.dns)
$(i)  Hopping range    : $(round(h.selector.range, digits = 6))
$(i)  Force hermitian  : $(h.forcehermitian)
$(i)  Coefficient      : $(h.coefficient)")
end

# External API #
"""
    onsite(o; sublats = missing, forcehermitian = true)

Create an `TightbindingModelTerm` that applies an onsite energy `o` to a `Lattice` when
creating a `Hamiltonian` with `hamiltonian`.

The onsite energy `o` can be a number, a matrix (preferably `SMatrix`) or a function of the
form `r -> ...` for a position-dependent onsite energy. If `sublats` is specified as a
sublattice name or tuple thereof, `onsite` is only applied to sublattices with said names.
If `forcehermitian` is true, the model will produce an hermitian Hamiltonian.

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
    dn cell jumps    : any
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
    `hopping`
"""
onsite(o; forcehermitian::Bool = true, kw...) =
    TightbindingModel(OnsiteTerm(o, selectonsites(;kw...), 1, forcehermitian))

"""
    hopping(h; sublats = missing, range = 1, dn = missing, forcehermitian = true)

Create an `TightbindingModelTerm` that applies a hopping `h` to a `Lattice` when
creating a `Hamiltonian` with `hamiltonian`.

The maximum distance between coupled sites is given by `range::Real`. If a cell distance
`dn::NTuple{L,Int}` or distances `dn::NTuple{M,NTuple{L,Int}}` are given, only unit cells
at that distance will be coupled.

The hopping amplitude `h` can be a number, a matrix (preferably `SMatrix`) or a function
of the form `(r, dr) -> ...` for a position-dependent hopping (`r` is the bond center,
and `dr` the bond vector). If `sublats` is specified as a sublattice name pair, or tuple
thereof, `hopping` is only applied between sublattices with said names. If `forcehermitian`
is true, the model will produce an hermitian Hamiltonian.

The dimension of `h::AbstractMatrix` must match the orbital dimension of applicable
sublattices (see also `orbitals` option for `hamiltonian`). If `h::Number` it will be
treated as `h * I` (proportional to identity matrix) when applied to multiorbital
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
    dn cell jumps    : ([1, 2], [0, 0])
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
    `onsite`
"""
hopping(h; forcehermitian::Bool = true, kw...) =
    TightbindingModel(HoppingTerm(h, selecthoppings(;kw...), 1, forcehermitian))


Base.:*(x, o::OnsiteTerm) =
    OnsiteTerm(o.o, o.selector, x * o.coefficient, o.forcehermitian)
Base.:*(x, t::HoppingTerm) =
    HoppingTerm(t.h, t.selector, x * t.coefficient, t.forcehermitian)
Base.:*(t::TightbindingModelTerm, x) = x * t
Base.:-(t::TightbindingModelTerm) = (-1) * t

LinearAlgebra.ishermitian(t::OnsiteTerm) = t.forcehermitian
LinearAlgebra.ishermitian(t::HoppingTerm) = t.forcehermitian

# Base.:+(t1::TightbindingModelTerm, t2::TightbindingModelTerm) = TightbindingModel((t1, t2))
# Base.:-(t1::TightbindingModelTerm, t2::TightbindingModelTerm) = TightbindingModel((t1, -t2))

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
    offdiagonal(model, hams::Hamiltonian...)

Build a restricted version of `model` that applies strictly between hamiltonians `hams`,
but not within each of them.
"""
offdiagonal(m::TightbindingModel{N}, hams) where {N} =
    TightbindingModel(ntuple(i -> offdiagonal(m.terms[i], hams), Val(N)))

function offdiagonal(m::TightbindingModel{N}, hams::Tuple{Vararg{<:Hamiltonian}}) where {N}
    sranges = sublatranges(nsublats...)
end

struct NondiagonalTerm{T<:HoppingTerm,S<:Tuple{Vararg{UnitRange{Int}}}} <: AbstractHoppingTerm
    term::T
    sublatranges::S
end

function nondiagonal(m::TightbindingModel{N}, nsublats::Tuple{Vararg{Int}}) where {N}
    sranges = sublatranges(nsublats...)
    return TightbindingModel(ntuple(i -> NondiagonalTerm(m.terms[i], sranges), Val(N)))
end


sublatranges(i::Int, is::Int...) = _sublatranges((1:i,), is...)
_sublatranges(rs::Tuple, i::Int, is...) = _sublatranges((rs..., last(last(rs)) + 1: last(last(rs)) + i), is...)
_sublatranges(rs::Tuple) = rs

function Base.show(io::IO, h::NondiagonalTerm)
    i = get(io, :indent, "")
    show(io, h.term)
    print(io,"\n$(i)  Nondiagonal only : $(h.sublatranges)")
end

Base.:*(x, t::NondiagonalTerm) = NondiagonalTerm(x * t.term)

LinearAlgebra.ishermitian(t::NondiagonalTerm) = ishermitian(t.term)

function sublats(t::NondiagonalTerm, lat::AbstractLattice)
    s = sublats(t.term, lat)
    sr = t.sublatranges
    filter!(spair ->  findblock(first(spair), sr) != findblock(last(spair), sr), s)
    return s
end

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

_checkmodelorbs(nt::NondiagonalTerm, orbs, lat) = _checkmodelorbs(nt.term, orbs, lat)

_checkmodelorbs(s::SMatrix, m, n = m) =
    size(s) == (m, n) || @warn("Possible dimension mismatch between model and Hamiltonian. Did you correctly specify the `orbitals` in hamiltonian?")

_checkmodelorbs(s::Number, m, n = m) = _checkmodelorbs(SMatrix{1,1}(s), m, n)