#######################################################################
# Hamiltonian
#######################################################################
struct HamiltonianHarmonic{L,M,A<:AbstractMatrix{M}}
    dn::SVector{L,Int}
    h::A
end

HamiltonianHarmonic{L,M,A}(dn::SVector{L,Int}, n::Int, m::Int) where {L,M,A<:SparseMatrixCSC{M}} =
    HamiltonianHarmonic(dn, sparse(Int[], Int[], M[], n, m))

HamiltonianHarmonic{L,M,A}(dn::SVector{L,Int}, n::Int, m::Int) where {L,M,A<:Matrix{M}} =
    HamiltonianHarmonic(dn, zeros(M, n, m))

struct Hamiltonian{LA<:AbstractLattice,L,M,A<:AbstractMatrix,
                   H<:HamiltonianHarmonic{L,M,A},F<:Union{Missing,Field},
                   O<:Tuple{Vararg{Tuple{Vararg{NameType}}}}} <: AbstractArray{A,L}
    lattice::LA
    harmonics::Vector{H}
    field::F
    orbitals::O
end

function Hamiltonian(lat, hs::Vector{H}, field, orbs, n::Int, m::Int) where {L,M,H<:HamiltonianHarmonic{L,M}}
    sort!(hs, by = h -> abs.(h.dn))
    if isempty(hs) || !iszero(first(hs).dn)
        pushfirst!(hs, H(zero(SVector{L,Int}), empty_sparse(M, n, m)))
    end
    return Hamiltonian(lat, hs, field, orbs)
end

Base.show(io::IO, ham::Hamiltonian) = show(io, MIME("text/plain"), ham)
function Base.show(io::IO, ::MIME"text/plain", ham::Hamiltonian)
    i = get(io, :indent, "")
    print(io, i, summary(ham), "\n",
"$i  Bloch harmonics  : $(length(ham.harmonics)) ($(displaymatrixtype(ham)))
$i  Harmonic size    : $((n -> "$n × $n")(nsites(ham)))
$i  Orbitals         : $(displayorbitals(ham))
$i  Element type     : $(displayelements(ham))
$i  Onsites          : $(nonsites(ham))
$i  Hoppings         : $(nhoppings(ham))
$i  Coordination     : $(nhoppings(ham) / nsites(ham))")
    ioindent = IOContext(io, :indent => string("  "))
    issuperlattice(ham.lattice) && print(ioindent, "\n", ham.lattice.supercell)
end

Base.show(io::IO, h::HamiltonianHarmonic) = show(io, MIME("text/plain"), h)
Base.show(io::IO, ::MIME"text/plain", h::HamiltonianHarmonic{L,M}) where {L,M} = print(io,
"HamiltonianHarmonic{$L,$(eltype(M))} : Bloch harmonic of $(L)D Hamiltonian
  Harmonic type   : $(displaymatrixtype(typeof(h.h)))
  Harmonic size   : $((n -> "$n × $n")(nsites(h)))
  Cell distance   : $(Tuple(h.dn))
  Element type    : $(displayelements(M))
  Elements        : $(_nnz(h.h))")

Base.summary(::Hamiltonian{LA}) where {E,L,LA<:Lattice{E,L}} =
    "Hamiltonian{<:Lattice} : $(L)D Hamiltonian on a $(L)D Lattice in $(E)D space"

Base.summary(::Hamiltonian{LA}) where {E,L,T,L´,LA<:Superlattice{E,L,T,L´}} =
    "Hamiltonian{<:Superlattice} : $(L)D Hamiltonian on a $(L´)D Superlattice in $(E)D space"

matrixtype(::Hamiltonian{LA,L,M,A}) where {LA,L,M,A} = A
displaymatrixtype(h::Hamiltonian) = displaymatrixtype(matrixtype(h))
displaymatrixtype(::Type{<:SparseMatrixCSC}) = "SparseMatrixCSC, sparse"
displaymatrixtype(::Type{<:Array}) = "Matrix, dense"
displaymatrixtype(A::Type{<:AbstractArray}) = string(A)
displayelements(h::Hamiltonian) = displayelements(blocktype(h))
displayelements(::Type{S}) where {N,T,S<:SMatrix{N,N,T}} = "$N × $N blocks ($T)"
displayelements(::Type{T}) where {T} = "scalar ($T)"
displayorbitals(h::Hamiltonian) =
    replace(replace(string(h.orbitals), "Symbol(\"" => ":"), "\")" => "")

# Internal API #

# find SVector type that can hold all orbital amplitudes in any lattice sites
orbitaltype(orbs, type::Type{Tv} = Complex{T}) where {E,L,T,Tv} =
    _orbitaltype(SVector{1,Tv}, orbs...)
_orbitaltype(::Type{S}, ::NTuple{D,NameType}, os...) where {N,Tv,D,S<:SVector{N,Tv}} =
    (M = max(N,D); _orbitaltype(SVector{M,Tv}, os...))
_orbitaltype(t::Type{SVector{N,Tv}}) where {N,Tv} = t
_orbitaltype(t::Type{SVector{1,Tv}}) where {Tv} = Tv

# find SMatrix type that can hold all matrix elements between lattice sites
blocktype(orbs, type::Type{Tv} = Complex{T}) where {E,L,T,Tv} =
    _blocktype(orbitaltype(orbs, Tv))
_blocktype(::Type{S}) where {N,Tv,S<:SVector{N,Tv}} = SMatrix{N,N,Tv,N*N}
_blocktype(::Type{S}) where {S<:Number} = S

blocktype(h::Hamiltonian{LA,L,M}) where {LA,L,M} = M

function nhoppings(ham::Hamiltonian)
    count = 0
    for h in ham.harmonics
        count += iszero(h.dn) ? (_nnz(h.h) - _nnzdiag(h.h)) : _nnz(h.h)
    end
    return count
end

function nonsites(ham::Hamiltonian)
    count = 0
    for h in ham.harmonics
        iszero(h.dn) && (count += _nnzdiag(h.h))
    end
    return count
end

_nnz(h::SparseMatrixCSC) = nnz(h)
_nnz(h::Matrix) = count(!iszero, h)

function _nnzdiag(s::SparseMatrixCSC)
    count = 0
    rowptrs = rowvals(s)
    for col in 1:size(s,2)
        for ptr in nzrange(s, col)
            rowptrs[ptr] == col && (count += 1; break)
        end
    end
    return count
end
_nnzdiag(s::Matrix) = count(!iszero, s[i,i] for i in 1:minimum(size(s)))

nsites(h::Hamiltonian) = isempty(h.harmonics) ? 0 : nsites(first(h.harmonics))
nsites(h::HamiltonianHarmonic) = size(h.h, 1)

sanitize_orbs(os::NTuple{M,Union{Tuple,Val}}, names::NTuple{N}) where {N,M} =
    ntuple(n -> n > M ? (:a,) : sanitize_orbs(os[n]), Val(N))
sanitize_orbs(os::NTuple{M,Pair}, names::NTuple{N}) where {N,M} =
    ntuple(Val(N)) do n
        for m in 1:M
            first(os[m]) == names[n] && return sanitize_orbs(os[m])
        end
        return (:a,)
    end
sanitize_orbs(os::NTuple{M,Union{NameType,Integer}}, names::NTuple{N}) where {M,N} =
    (ont = nametype.(os); ntuple(n -> ont , Val(N)))
sanitize_orbs(o::Union{NameType,Integer,Pair,Val}, names) =
    sanitize_orbs((_ -> o).(names), names)
sanitize_orbs(o::Missing, names) = sanitize_orbs((:a,), names)

sanitize_orbs(o::Integer) = nametype(o)
sanitize_orbs(o::NameType) = o
sanitize_orbs(o::Val{N}) where {N} = ntuple(_ -> :a, Val(N))
sanitize_orbs(o::Tuple) = sanitize_orbs.(o)
sanitize_orbs(p::Pair) = sanitize_orbs(last(p))

# External API #
"""
    hamiltonian(lat[, model]; orbitals, field, type)

Create a `Hamiltonian` by additively applying `model::TighbindingModel` to the lattice `lat`
(see `hopping` and `onsite` for details on building tightbinding models).

The number of orbitals on each sublattice can be specified by the keyword `orbitals`
(otherwise all sublattices have one orbital by default). The following, and obvious
combinations, are possible formats for the `orbitals` keyword:

    orbitals = :a                # all sublattices have 1 orbital named :a
    orbitals = (:a,)             # same as above
    orbitals = (:a, :b, 3)       # all sublattices have 3 orbitals named :a and :b and :3
    orbitals = ((:a, :b), (:c,)) # first sublattice has 2 orbitals, second has one
    orbitals = ((:a, :b), :c)    # same as above
    orbitals = (Val(2), Val(1))  # same as above, with automatic names
    orbitals = (:A => (:a, :b), :D => :c) # sublattice :A has two orbitals, :D and rest have one
    orbitals = :D => Val(4)      # sublattice :D has four orbitals, rest have one

The matrix sizes of tightbinding `model` must match the orbitals specified. Internally, we
define a block size `N = max(num_orbitals)`. If `N = 1` (all sublattices with one orbital)
the the Hamiltonian element type is `type`. Otherwise it is `SMatrix{N,N,type}` blocks,
padded with the necessary zeros as required. Keyword `type` is `Complex{T}` by default,
where `T` is the number type of `lat`.

Advanced use: if a `field = f(r,dr,h)` function is given, it will modify the hamiltonian
element `h` operating on sites `r₁` and `r₂`, where `r = (r₁ + r₂)/2` and `dr = r₂ - r₁`.
In combination with `supercell`, it allows to do matrix-free operations including position-
dependent perturbations (e.g. disorder, gauge fields).

    h(ϕ₁, ϕ₂, ...)
    h((ϕ₁, ϕ₂, ...))

Build the Bloch Hamiltonian matrix `bloch(h, (ϕ₁, ϕ₂, ...))` of a `h::Hamiltonian` on an
`L`D lattice. (See also `bloch!` for a non-allocating version of `bloch`.)

    hamiltonian(lat, [model,] funcmodel::Function; kw...)

For a function of the form `funcmodel(;params...)::TightbindingModel`, produce a
`h::ParametricHamiltonian` that efficiently generates a `Hamiltonian` with model `model +
funcmodel(;params...)` when calling it as in `h(;params...)` (using specific parameters as
keyword arguments `params`). Additionally, `h(ϕ₁, ϕ₂, ...; params...)` generates the
corresponding Bloch Hamiltonian matrix (equivalent to `h(;params...)(ϕ₁, ϕ₂, ...)`).

It's important to note that `params` keywords in the definition of `funcmodel` must have
default values, as in `model(;o = 1) = onsite(o)`.

    lat |> hamiltonian([func, model]; kw...)

Functional form of `hamiltonian`, equivalent to `hamiltonian(lat, ...; ...)`

# Indexing

Indexing into a Hamiltonian `h` works as follows. Access the `HamiltonianHarmonic` matrix
at a given `dn::NTuple{L,Int}` with `h[dn...]`. Assign `v` into element `(i,j)` of said
matrix with `h[dn...][i,j] = v`. Broadcasting with vectors of indices `is` and `js` is
supported, `h[dn...][is, js] = v_matrix`.

To add an empty harmonic with a given `dn::NTuple{L,Int}`, do `push!(h, dn)`. To delete it,
do `deleteat!(h, dn)`.

# Examples
```jldoctest
julia> h = hamiltonian(LatticePresets.honeycomb(), hopping(@SMatrix[1 2; 3 4], range = 1/√3), orbitals = Val(2))
Hamiltonian{<:Lattice} : 2D Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 5 (SparseMatrixCSC, sparse)
  Harmonic size    : 2 × 2
  Orbitals         : ((:a, :a), (:a, :a))
  Element type     : 2 × 2 blocks (Complex{Float64})
  Onsites          : 0
  Hoppings         : 6
  Coordination     : 3.0

julia> push!(h, (3,3)) # Adding a new Hamiltonian harmonic (if not already present)
Hamiltonian{<:Lattice} : 2D Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 6 (SparseMatrixCSC, sparse)
  Harmonic size    : 2 × 2
  Orbitals         : ((:a, :a), (:a, :a))
  Element type     : 2 × 2 blocks (Complex{Float64})
  Onsites          : 0
  Hoppings         : 6
  Coordination     : 3.0

julia> h[3,3][1,1] = @SMatrix[1 2; 2 1]; h[3,3] # element assignment
2×2 SparseArrays.SparseMatrixCSC{StaticArrays.SArray{Tuple{2,2},Complex{Float64},2,4},Int64} with 1 stored entry:
  [1, 1]  =  [1.0+0.0im 2.0+0.0im; 2.0+0.0im 1.0+0.0im]

julia> h[3,3][[1,2],[1,2]] .= rand(SMatrix{2,2,Float64}, 2, 2) # Broadcast assignment
2×2 view(::SparseArrays.SparseMatrixCSC{StaticArrays.SArray{Tuple{2,2},Complex{Float64},2,4},Int64}, [1, 2], [1, 2]) with eltype StaticArrays.SArray{Tuple{2,2},Complex{Float64},2,4}:
 [0.271152+0.0im 0.921417+0.0im; 0.138212+0.0im 0.525911+0.0im]  [0.444284+0.0im 0.280035+0.0im; 0.565106+0.0im 0.121869+0.0im]
 [0.201126+0.0im 0.912446+0.0im; 0.372099+0.0im 0.931358+0.0im]  [0.883422+0.0im 0.874016+0.0im; 0.296095+0.0im 0.995861+0.0im]

julia> hopfunc(;k = 0) = hopping(k); hamiltonian(LatticePresets.square(), onsite(1) + hopping(2), hopfunc) # Parametric Hamiltonian
Parametric Hamiltonian{<:Lattice} : 2D Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 5 (SparseMatrixCSC, sparse)
  Harmonic size    : 1 × 1
  Orbitals         : ((:a,),)
  Elements         : scalars (Complex{Float64})
  Onsites          : 1
  Hoppings         : 4
  Coordination     : 4.0

```
"""
hamiltonian(lat, ts...; orbitals = missing, kw...) =
    _hamiltonian(lat, sanitize_orbs(orbitals, lat.unitcell.names), ts...; kw...)
_hamiltonian(lat::AbstractLattice, orbs; kw...) = _hamiltonian(lat, orbs, TightbindingModel(); kw...)
_hamiltonian(lat::AbstractLattice, orbs, f::Function; kw...) = _hamiltonian(lat, orbs, TightbindingModel(), f; kw...)
_hamiltonian(lat::AbstractLattice, orbs, m::TightbindingModel; type::Type = Complex{numbertype(lat)}, kw...) =
    hamiltonian_sparse(blocktype(orbs, type), lat, orbs, m; kw...)
_hamiltonian(lat::AbstractLattice, orbs, m::TightbindingModel, f::Function;
            type::Type = Complex{numbertype(lat)}, kw...) =
    parametric_hamiltonian(blocktype(orbs, type), lat, orbs, m, f; kw...)

hamiltonian(t::TightbindingModel...; kw...) =
    z -> hamiltonian(z, t...; kw...)
hamiltonian(f::Function, t::TightbindingModel...; kw...) =
    z -> hamiltonian(z, f, t...; kw...)
hamiltonian(h::Hamiltonian) =
    z -> hamiltonian(z, h)

(h::Hamiltonian)(phases...) = bloch(h, phases...)

Base.Matrix(h::Hamiltonian) = Hamiltonian(h.lattice, Matrix.(h.harmonics), h.field, h.orbitals)
Base.Matrix(h::HamiltonianHarmonic) = HamiltonianHarmonic(h.dn, Matrix(h.h))

Base.copy(h::Hamiltonian) = Hamiltonian(h.lattice, copy.(h.harmonics), h.field)
Base.copy(h::HamiltonianHarmonic) = HamiltonianHarmonic(h.dn, copy(h.h))

Base.size(h::Hamiltonian, n) = size(first(h.harmonics).h, n)
Base.size(h::Hamiltonian) = size(first(h.harmonics).h)
Base.size(h::HamiltonianHarmonic, n) = size(h.h, n)
Base.size(h::HamiltonianHarmonic) = size(h.h)

bravais(h::Hamiltonian) = bravais(h.lattice)

issemibounded(h::Hamiltonian) = issemibounded(h.lattice)

# Indexing #

Base.push!(h::Hamiltonian{<:Any,L}, dn::NTuple{L,Int}) where {L} = push!(h, SVector(dn...))
Base.push!(h::Hamiltonian{<:Any,L}, dn::Vararg{Int,L}) where {L} = push!(h, SVector(dn...))
function Base.push!(h::Hamiltonian{<:Any,L,M,A}, dn::SVector{L,Int}) where {L,M,A}
    for hh in h.harmonics
        hh.dn == dn && return hh
    end
    hh = HamiltonianHarmonic{L,M,A}(dn, size(h)...)
    push!(h.harmonics, hh)
    return h
end

@inline function Base.getindex(h::Hamiltonian{<:Any,L}, dn::Vararg{Int,L}) where {L}
    dnv = SVector(dn...)
    nh = findfirst(hh -> hh.dn == dnv, h.harmonics)
    nh === nothing && throw(BoundsError(h, dn))
    return h.harmonics[nh].h
end

Base.deleteat!(h::Hamiltonian{<:Any,L}, dn::Vararg{Int,L}) where {L} =
    deleteat!(h, toSVector(dn))
Base.deleteat!(h::Hamiltonian{<:Any,L}, dn::NTuple{L,Int}) where {L} =
    deleteat!(h, toSVector(dn))
function Base.deleteat!(h::Hamiltonian{<:Any,L}, dn::SVector{L,Int}) where {L}
    nh = findfirst(hh -> hh.dn == SVector(dn...), h.harmonics)
    nh === nothing || deleteat!(h.harmonics, nh)
    return h
end

Base.isassigned(h::Hamiltonian{<:Any,L}, dn::Vararg{Int,L}) where {L} = isassigned(h, dn)
function Base.isassigned(h::Hamiltonian{<:Any,L}, dn::NTuple{L,Int}) where {L}
    dnv = SVector(dn...)
    nh = findfirst(hh -> hh.dn == dnv, h.harmonics)
    return nh !== nothing
end

#######################################################################
# auxiliary types
#######################################################################
struct IJV{L,M}
    dn::SVector{L,Int}
    i::Vector{Int}
    j::Vector{Int}
    v::Vector{M}
end

struct IJVBuilder{L,M,E,T,O,LA<:AbstractLattice{E,L,T}}
    lat::LA
    orbs::O
    ijvs::Vector{IJV{L,M}}
    kdtrees::Vector{KDTree{SVector{E,T},Euclidean,T}}
end

IJV{L,M}(dn::SVector{L} = zero(SVector{L,Int})) where {L,M} =
    IJV(dn, Int[], Int[], M[])

function IJVBuilder{M}(lat::AbstractLattice{E,L,T}, orbs) where {E,L,T,M}
    ijvs = IJV{L,M}[]
    kdtrees = Vector{KDTree{SVector{E,T},Euclidean,T}}(undef, nsublats(lat))
    return IJVBuilder(lat, orbs, ijvs, kdtrees)
end

function Base.getindex(b::IJVBuilder{L,M}, dn::SVector{L2,Int}) where {L,L2,M}
    L == L2 || throw(error("Tried to apply an $L2-dimensional model to an $L-dimensional lattice"))
    for e in b.ijvs
        e.dn == dn && return e
    end
    e = IJV{L,M}(dn)
    push!(b.ijvs, e)
    return e
end

Base.length(h::IJV) = length(h.i)
Base.isempty(h::IJV) = length(h) == 0
Base.copy(h::IJV) = IJV(h.dn, copy(h.i), copy(h.j), copy(h.v))

function Base.resize!(h::IJV, n)
    resize!(h.i, n)
    resize!(h.j, n)
    resize!(h.v, n)
    return h
end

Base.push!(h::IJV, (i, j, v)) = (push!(h.i, i); push!(h.j, j); push!(h.v, v))

#######################################################################
# hamiltonian_sparse
#######################################################################
function hamiltonian_sparse(::Type{M}, lat::AbstractLattice{E,L}, orbs, model; field = missing) where {E,L,M}
    builder = IJVBuilder{M}(lat, orbs)
    applyterms!(builder, terms(model)...)
    HT = HamiltonianHarmonic{L,M,SparseMatrixCSC{M,Int}}
    n = nsites(lat)
    harmonics = HT[HT(e.dn, sparse(e.i, e.j, e.v, n, n)) for e in builder.ijvs if !isempty(e)]
    return Hamiltonian(lat, harmonics, Field(field, lat), orbs, n, n)
end

applyterms!(builder, terms...) = foreach(term -> applyterm!(builder, term), terms)

function applyterm!(builder::IJVBuilder{L,M}, term::OnsiteTerm) where {L,M}
    lat = builder.lat
    for s in sublats(term, lat)
        is = siterange(lat, s)
        dn0 = zero(SVector{L,Int})
        ijv = builder[dn0]
        offset = lat.unitcell.offsets[s]
        for i in is
            r = lat.unitcell.sites[i]
            vs = orbsized(term(r,r), builder.orbs[s])
            v = padtotype(vs, M)
            term.forcehermitian ? push!(ijv, (i, i, 0.5 * (v + v'))) : push!(ijv, (i, i, v))
        end
    end
    return nothing
end

function applyterm!(builder::IJVBuilder{L,M}, term::HoppingTerm) where {L,M}
    checkinfinite(term)
    lat = builder.lat
    for (s1, s2) in sublats(term, lat)
        is, js = siterange(lat, s1), siterange(lat, s2)
        dns = dniter(term.dns, Val(L))
        for dn in dns
            addadjoint = term.forcehermitian
            foundlink = false
            ijv = builder[dn]
            addadjoint && (ijvc = builder[negative(dn)])
            for j in js
                sitej = lat.unitcell.sites[j]
                rsource = sitej - lat.bravais.matrix * dn
                itargets = targets(builder, term.range, rsource, s1)
                for i in itargets
                    isselfhopping((i, j), (s1, s2), dn) && continue
                    foundlink = true
                    rtarget = lat.unitcell.sites[i]
                    r, dr = _rdr(rsource, rtarget)
                    vs = orbsized(term(r, dr), builder.orbs[s1], builder.orbs[s2])
                    v = padtotype(vs, M)
                    if addadjoint
                        v *= redundancyfactor(dn, (s1, s2), term)
                        push!(ijv, (i, j, v))
                        push!(ijvc, (j, i, v'))
                    else
                        push!(ijv, (i, j, v))
                    end
                end
            end
            foundlink && acceptcell!(dns, dn)
        end
    end
    return nothing
end

orbsized(m, orbs) = orbsized(m, orbs, orbs)
orbsized(m, o1::NTuple{D1}, o2::NTuple{D2}) where {D1,D2} =
    SMatrix{D1,D2}(m)
orbsized(m::Number, o1::NTuple{1}, o2::NTuple{1}) = m

dniter(dns::Missing, ::Val{L}) where {L} = BoxIterator(zero(SVector{L,Int}))
dniter(dns, ::Val) = dns

function targets(builder, range::Real, rsource, s1)
    if !isassigned(builder.kdtrees, s1)
        sites = view(builder.lat.unitcell.sites, siterange(builder.lat, s1))
        (builder.kdtrees[s1] = KDTree(sites))
    end
    targets = inrange(builder.kdtrees[s1], rsource, range)
    targets .+= builder.lat.unitcell.offsets[s1]
    return targets
end

targets(builder, range::Missing, rsource, s1) = eachindex(builder.lat.sublats[s1].sites)

checkinfinite(term) = term.dns === missing && (term.range === missing || !isfinite(term.range)) &&
    throw(ErrorException("Tried to implement an infinite-range hopping on an unbounded lattice"))

isselfhopping((i, j), (s1, s2), dn) = i == j && s1 == s2 && iszero(dn)

# Avoid double-counting hoppings when adding adjoint
redundancyfactor(dn, ss, term) =
    isnotredundant(dn, term) || isnotredundant(ss, term) ? 1.0 : 0.5
# (i,j,dn) and (j,i,-dn) will not both be added if any of the following is true
isnotredundant(dn::SVector, term) = term.dns !== missing && !iszero(dn)
isnotredundant((s1, s2)::Tuple{Int,Int}, term) = term.sublats !== missing && s1 != s2

#######################################################################
# unitcell/supercell for Hamiltonians
#######################################################################
function supercell(ham::Hamiltonian, args...; kw...)
    slat = supercell(ham.lattice, args...; kw...)
    return Hamiltonian(slat, ham.harmonics, ham.field, ham.orbitals)
end

function unitcell(ham::Hamiltonian{<:Lattice}, args...; kw...)
    sham = supercell(ham, args...; kw...)
    return unitcell(sham)
end

function unitcell(ham::Hamiltonian{LA,L,Tv}) where {E,L,T,L´,Tv,LA<:Superlattice{E,L,T,L´}}
    lat = ham.lattice
    sc = lat.supercell
    mapping = OffsetArray{Int}(undef, sc.sites, sc.cells.indices...) # store supersite indices newi
    mapping .= 0
    foreach_supersite((s, oldi, olddn, newi) -> mapping[oldi, Tuple(olddn)...] = newi, lat)
    dim = nsites(sc)
    B = blocktype(ham)
    harmonic_builders = HamiltonianHarmonic{L´,Tv,SparseMatrixBuilder{B}}[]
    pinvint = pinvmultiple(sc.matrix)
    foreach_supersite(lat) do s, source_i, source_dn, newcol
        for oldh in ham.harmonics
            rows = rowvals(oldh.h)
            vals = nonzeros(oldh.h)
            target_dn = source_dn + oldh.dn
            super_dn = new_dn(target_dn, pinvint)
            wrapped_dn = wrap_dn(target_dn, super_dn, sc.matrix)
            newh = get_or_push!(harmonic_builders, super_dn, dim)
            for p in nzrange(oldh.h, source_i)
                target_i = rows[p]
                # check: wrapped_dn could exit bounding box along non-periodic direction
                checkbounds(Bool, mapping, target_i, Tuple(wrapped_dn)...) || continue
                newrow = mapping[target_i, Tuple(wrapped_dn)...]
                val = applyfield(ham.field, vals[p], target_i, source_i, source_dn)
                iszero(newrow) || pushtocolumn!(newh.h, newrow, val)
            end
        end
        foreach(h -> finalisecolumn!(h.h), harmonic_builders)
    end
    harmonics = [HamiltonianHarmonic(h.dn, sparse(h.h)) for h in harmonic_builders]
    unitlat = unitcell(lat)
    field = ham.field
    orbs = ham.orbitals
    return Hamiltonian(unitlat, harmonics, field, orbs)
end

function get_or_push!(hs::Vector{HamiltonianHarmonic{L,Tv,SparseMatrixBuilder{B}}}, dn, dim) where {L,Tv,B}
    for h in hs
        h.dn == dn && return h
    end
    newh = HamiltonianHarmonic(dn, SparseMatrixBuilder{B}(dim, dim))
    push!(hs, newh)
    return newh
end

#######################################################################
# parametric hamiltonian
#######################################################################
struct ParametricHamiltonian{H,F,E,T}
    base::H             # Hamiltonian before applying parametrized model
    hamiltonian::H      # Hamiltonian to update that includes parametrized model
    pointers::Vector{Vector{Tuple{Int,SVector{E,T},SVector{E,T}}}} # val pointers to modify
    f::F                                                           # by f on each harmonic
end

Base.show(io::IO, pham::ParametricHamiltonian) = print(io, "Parametric ", pham.hamiltonian)

function parametric_hamiltonian(::Type{M}, lat::AbstractLattice{E,L,T}, orbs, model, f::F;
                                field = missing) where {M,E,L,T,F}
    builder = IJVBuilder{M}(lat, orbs)
    applyterms!(builder, terms(model)...)
    nels = length.(builder.ijvs) # element counters for each harmonic
    model_f = f()
    applyterms!(builder, terms(model_f)...)
    padright!(nels, 0, length(builder.ijvs)) # in case new harmonics where added
    nels_f = length.(builder.ijvs) # element counters after adding f model
    empties = isempty.(builder.ijvs)
    deleteat!(builder.ijvs, empties)
    deleteat!(nels, empties)
    deleteat!(nels_f, empties)

    base_ijvs = copy.(builder.ijvs) # ijvs for ham without f, but with structural zeros
    zeroM = zero(M)
    for (ijv, nel, nel_f) in zip(base_ijvs, nels, nels_f), p in nel+1:nel_f
        ijv.v[p] = zeroM
    end

    HT = HamiltonianHarmonic{L,M,SparseMatrixCSC{M,Int}}
    n = nsites(lat)
    base_harmonics = HT[HT(e.dn, sparse(e.i, e.j, e.v, n, n)) for e in base_ijvs]
    harmonics = HT[HT(e.dn, sparse(e.i, e.j, e.v, n, n)) for e in builder.ijvs]
    pointers = [getpointers(harmonics[k].h, builder.ijvs[k], nels[k], lat) for k in eachindex(harmonics)]
    base_h = Hamiltonian(lat, base_harmonics, missing, orbs, n, n)
    h = Hamiltonian(lat, harmonics, Field(field, lat), orbs, n, n)
    return ParametricHamiltonian(base_h, h, pointers, f)
end

function getpointers(h::SparseMatrixCSC, ijv, eloffset, lat::AbstractLattice{E,L,T}) where {E,L,T}
    rows = rowvals(h)
    sites = lat.unitcell.sites
    pointers = Tuple{Int,SVector{E,T},SVector{E,T}}[] # (pointer, r, dr)
    nelements = length(ijv)
    for k in eloffset+1:nelements
        row = ijv.i[k]
        col = ijv.j[k]
        for ptr in nzrange(h, col)
            if row == rows[ptr]
                r, dr = _rdr(sites[col], sites[row]) # _rdr(source, target)
                push!(pointers, (ptr, r, dr))
                break
            end
        end
    end
    unique!(first, pointers) # adjoint duplicates lead to repeated pointers... remove.
    return pointers
end

function initialize!(ph::ParametricHamiltonian{H}) where {LA,L,M,A<:SparseMatrixCSC,H<:Hamiltonian{LA,L,M,A}}
    for (bh, h, prdrs) in zip(ph.base.harmonics, ph.hamiltonian.harmonics, ph.pointers)
        vals = nonzeros(h.h)
        vals_base = nonzeros(bh.h)
        for (p,_,_) in prdrs
            vals[p] = vals_base[p]
        end
    end
    return nothing
end

function applyterm!(ph::ParametricHamiltonian{H}, term::TightbindingModelTerm)  where {LA,L,M,A<:SparseMatrixCSC,H<:Hamiltonian{LA,L,M,A}}
    for (h, prdrs) in zip(ph.hamiltonian.harmonics, ph.pointers)
        vals = nonzeros(h.h)
        for (p, r, dr) in prdrs
            v = term(r, dr) # should perhaps be v = orbsized(term(r, dr), orb1, orb2)
            vals[p] += padtotype(v, M)
        end
    end
    return nothing
end

# API #

function (ph::ParametricHamiltonian)(;kw...)
    isempty(kw) && return ph.hamiltonian
    model = ph.f(;kw...)
    initialize!(ph)
    foreach(term -> applyterm!(ph, term), terms(model))
    return ph.hamiltonian
end

(ph::ParametricHamiltonian)(arg, args...; kw...) = ph(;kw...)(arg, args...)

Base.Matrix(h::ParametricHamiltonian) =
    ParametricHamiltonian(Matrix(h.base), Matrix(h.hamiltonian), h.pointers, h.f)

Base.copy(h::ParametricHamiltonian) =
    ParametricHamiltonian(copy(h.base), copy(h.hamiltonian), copy(h.pointers), h.field)

Base.size(h::ParametricHamiltonian, n...) = size(h.hamiltonian, n...)

bravais(ph::ParametricHamiltonian) = bravais(ph.hamiltonian.lattice)

Base.eltype(::ParametricHamiltonian{H}) where {L,M,H<:Hamiltonian{L,M}} = M

#######################################################################
# Bloch routines
#######################################################################
struct SupercellBloch{L,T,H<:Hamiltonian{<:Superlattice}}
    hamiltonian::H
    phases::SVector{L,T}
end

Base.summary(h::SupercellBloch{L,T}) where {L,T} =
    "SupercellBloch{$L)}: Bloch Hamiltonian matrix lazily defined on an $(L)D supercell"

function Base.show(io::IO, sb::SupercellBloch)
    ioindent = IOContext(io, :indent => string("  "))
    print(io, summary(sb), "\n  Phases          : $(Tuple(sb.phases))\n")
    print(ioindent, sb.hamiltonian.lattice.supercell)
end

"""
    bloch!(matrix, h::Hamiltonian{<:Lattice}, ϕs::Real...)
    bloch!(matrix, h::Hamiltonian{<:Lattice}, ϕs::NTuple{L,Real})
    bloch!(matrix, h::Hamiltonian{<:Lattice}, ϕs::AbstractVector{Real})

Overwrite `matrix` with the Bloch Hamiltonian matrix of `h`, for the specified Bloch
phases `ϕs`. In terms of Bloch wavevector `k`, `ϕs = k * bravais(h)`. If all `ϕs` are
omitted, the intracell Hamiltonian is returned instead. If the Hamiltonian is defined on a
`Superlattice`, the evaluation of the Bloch Hamiltonian is deferred until it is used (e.g.
in a multiplication).

A suitable, non-initialized `matrix` can be obtained with `similar(h)`.

If `optimize!(h)` is called on a sparse Hamiltonian `h` before the first call to `bloch!`,
performance will increase by avoiding memory reshuffling.

# Examples
```
julia> LatticePresets.honeycomb() |> hamiltonian(onsite(1), hopping(2)) |> bloch!(.2,.3)
2×2 SparseArrays.SparseMatrixCSC{Complex{Float64},Int64} with 4 stored entries:
  [1, 1]  =  1.99001-0.199667im
  [2, 1]  =  1.96013-0.397339im
  [1, 2]  =  1.96013+0.397339im
  [2, 2]  =  1.99001-0.199667im
```

# See also:
    bloch, optimize!
"""
function bloch!(matrix::A, h::Hamiltonian{<:Lattice,L,M,A}, ϕs...) where {L,M,A}
    copy!(matrix, first(h.harmonics).h)
    return add_harmonics!(matrix, h, ϕs...)
end

bloch(h::Hamiltonian{<:Superlattice}, ϕs...) = SupercellBloch(h, toSVector(ϕs))

add_harmonics!(zerobloch, h::Hamiltonian, ϕs::Number...) =
    add_harmonics!(zerobloch, h, toSVector(ϕs))
add_harmonics!(zerobloch, h::Hamiltonian, ϕs::Tuple) =
    add_harmonics!(zerobloch, h, toSVector(ϕs))

add_harmonics!(zerobloch::A, h::Hamiltonian{<:Lattice,L,M,A}, ϕs::SVector{0}) where {L,M,A<:SparseMatrixCSC} =
    zerobloch
function add_harmonics!(zerobloch::A, h::Hamiltonian{<:Lattice,L,M,A}, ϕs::SVector{L}) where {L,M,A<:SparseMatrixCSC}
    for ns in 2:length(h.harmonics)
        hh = h.harmonics[ns]
        hhmatrix = hh.h
        ephi = cis(ϕs' * hh.dn)
        for col in 1:size(hhmatrix, 2)
            range = nzrange(hhmatrix, col)
            for ptr in range
                row = hhmatrix.rowval[ptr]
                zerobloch[row, col] = ephi * hhmatrix.nzval[ptr]
            end
        end
    end
    return zerobloch
end

function add_harmonics!(zerobloch::A, h::Hamiltonian{<:Lattice,L,M,A}, phases::SVector{L}) where {L,M,A<:Matrix}
    for ns in 2:length(h.harmonics)
        hh = h.harmonics[ns]
        ephi = cis(phases' * hh.dn)
        zerobloch .+= ephi .* hh.h
    end
    return zerobloch
end

"""
    bloch(h::Hamiltonian{<:Lattice}, ϕs::Real...)
    bloch(h::Hamiltonian{<:Lattice}, ϕs::NTuple{L,Real})
    bloch(h::Hamiltonian{<:Lattice}, ϕs::AbstractVector{Real})

Build the Bloch Hamiltonian matrix of `h`, for the specified Bloch phases `ϕs`. In terms of
Bloch wavevector `k`, `ϕs = k * bravais(h)`. If all `ϕs` are omitted, the intracell
Hamiltonian is returned instead. If the Hamiltonian is defined on a `Superlattice`, the
evaluation of the Bloch Hamiltonian is deferred until it is used (e.g. in a multiplication).

    h |> bloch(ϕs...)
    h(ϕs...)

Functional forms of `bloch`, equivalent to `bloch(h, ϕs...)`

This function allocates a new matrix on each call. For a non-allocating version of `bloch`,
see `bloch!`. If `optimize!(h)` is called on a sparse Hamiltonian `h` before the first call
to `bloch`, performance will increase by avoiding memory reshuffling.

    bloch(h::Hamiltonian{<:Superlattice}, ϕs...)

Build a `SupercellBloch` object that lazily implements the Bloch Hamiltonian in the
`Superlattice` without actually building the matrix (e.g. for matrix-free diagonalization).

# Examples
```
julia> LatticePresets.honeycomb() |> hamiltonian(onsite(1), hopping(2)) |> bloch(.2,.3)
2×2 SparseArrays.SparseMatrixCSC{Complex{Float64},Int64} with 4 stored entries:
  [1, 1]  =  1.99001-0.199667im
  [2, 1]  =  1.96013-0.397339im
  [1, 2]  =  1.96013+0.397339im
  [2, 2]  =  1.99001-0.199667im
```

# See also:
    bloch!
"""
bloch(phases...) = h -> bloch(h, phases...)
bloch(h::Hamiltonian{<:Lattice}, phases...) = bloch!(similar(h), h, phases...)

"""
    similar(h::Hamiltonian)

Create an uninitialized array of the same type of the Hamiltonian's matrix.
"""
Base.similar(h::Hamiltonian) = similar(h.harmonics[1].h)

"""
    optimize!(h::Hamiltonian)

Prepare a sparse Hamiltonian `h` to increase the performance of subsequent calls to
`bloch(h, ϕs...)` and `bloch!(matrix, h, ϕs...)` by minimizing memory reshufflings.

No optimization will be performed on non-sparse Hamiltonians, or those defined on
`Superlattice`s, for which Bloch Hamiltonians are lazily evaluated.

# See also:
    bloch, bloch!
"""
function optimize!(ham::Hamiltonian{<:Lattice,L,M,A}) where {LA,L,M,A<:SparseMatrixCSC}
    Tv = eltype(M)
    h0 = first(ham.harmonics)
    n, m = size(h0.h)
    iszero(h0.dn) || throw(ArgumentError("First Hamiltonian harmonic is not the fundamental"))
    nh = length(ham.harmonics)
    builder = SparseMatrixBuilder{M}(n, m)
    for col in 1:m
        for i in eachindex(ham.harmonics)
            h = ham.harmonics[i].h
            for p in nzrange(h, col)
                v = i == 1 ? nonzeros(h)[p] : zero(M)
                row = rowvals(h)[p]
                pushtocolumn!(builder, row, v, false) # skips repeated rows
            end
        end
        finalisecolumn!(builder)
    end
    ho = sparse(builder)
    copy!(h0.h, ho) # Inject new structural zeros into zero harmonics
    return ham
end

function optimize!(ham::Hamiltonian{<:Lattice,L,M,A}) where {LA,L,M,A<:AbstractMatrix}
    @warn "Hamiltonian is not sparse. Nothing changed."
    return ham
end

function optimize!(ham::Hamiltonian{<:Superlattice})
    @warn "Hamiltonian is defined on a Superlattice. Nothing changed."
    return ham
end

#######################################################################
# Flattened bloch
#######################################################################
# # More specific method for zerobloch with different eltype
# function optimized_zerobloch!(matrix::SparseMatrixCSC{<:Number}, h::Hamiltonian{<:Lattice,<:Any,<:SMatrix})
#     # h0 = first(h.harmonics).h
#     # if length(h0.nzval) != length(h.matrix.nzval) # first call, align first harmonic h0 with
#     #     copy!(h0.colptr, h.matrix.colptr)         # optimized h.matrix
#     #     copy!(h0.rowval, h.matrix.rowval)
#     #     copy!(h0.nzval,  h.matrix.nzval)
#     #     matrix === h.matrix || copy!(matrix, h.matrix)  # Also copy optimized h.matrix to matrix
#     # else  # if h.matrix, assume it's dirty and overwrite. Otherwise copy first harmonic, already optimized
#     #     matrix === h.matrix ? copy!(h.matrix.nzval, h0.nzval) : copy!(matrix, h0)
#     # end
#     # return matrix
# end

# function add_harmonics!(zerobloch::SparseMatrixCSC{<:Number}, h::Hamiltonian{<:Lattice,L,<:SMatrix}, ϕs::SVector{L}) where {L}

# end

# function blochflat!(matrix, h::Hamiltonian{<:Lattice,L,M,<:Matrix}, phases...) where {L,M<:SMatrix}
#     bloch!(h, phases...)
#     lat = h.lattice
#     offsets = flatoffsets(lat)
#     numorbs = numorbitals(lat)
#     for s2 in 1:nsublats(lat), s1 in 1:nsublats(lat)
#         offset1, offset2 = offsets[s1], offsets[s2]
#         norb1, norb2 = numorbs[s1], numorbs[s2]
#         for (m, j) in enumerate(siterange(lat, s2)), (n, i) in enumerate(siterange(lat, s1))
#             ioffset, joffset = offset1 + (n-1)*norb1, offset2 + (m-1)*norb2
#             el = h.matrix[i, j]
#             for sj in 1:norb2, si in 1:norb1
#                 matrix[ioffset + si, joffset + sj] = el[si, sj]
#             end
#         end
#     end
#     return matrix
# end

# function blochflat!(matrix, h::Hamiltonian{<:Lattice,L,<:Number}, phases...) where {L}
#     bloch!(h, phases...)
#     copy!(matrix, h.matrix)
#     return matrix
# end

# function blochflat!(h::Hamiltonian{<:Lattice,L,<:Number}, phases...) where {L}
#     bloch!(h, phases...)
#     return h.matrix
# end

# function blochflat(h::Hamiltonian{<:Lattice,L,M,<:Matrix}, phases...) where {L,M<:SMatrix}
#     dim = flatdim(h.lattice)
#     return blochflat!(similar(h.matrix, eltype(M), (dim, dim)), h, phases...)
# end

# function blochflat(h::Hamiltonian{<:Lattice,L,<:Number}, phases...) where {L}
#     bloch!(h, phases...)
#     return copy(h.matrix)
# end

# function blochflat(h::Hamiltonian{<:Lattice,L,M,<:SparseMatrixCSC}, phases...) where {L,M<:SMatrix}
#     bloch!(h, phases...)
#     lat = h.lattice
#     offsets = flatoffsets(lat)
#     numorbs = numorbitals(lat)
#     dim = flatdim(h.lattice)
#     builder = SparseMatrixBuilder{eltype(M)}(dim, dim)
#     for s2 in 1:nsublats(lat)
#         norb2 = numorbs[s2]
#         for col in siterange(lat, s2), sj in 1:norb2
#             for ptr in nzrange(h.matrix, col)
#                 row = rowvals(h.matrix)[ptr]
#                 val = nonzeros(h.matrix)[ptr]
#                 fo, s1 = flatoffset_sublat(lat, row, numorbs, offsets)
#                 norb1 = numorbs[s1]
#                 for si in 1:norb1
#                     flatrow = fo + si
#                     pushtocolumn!(builder, flatrow, val[si, sj])
#                 end
#             end
#             finalisecolumn!(builder)
#         end
#     end
#     matrix = sparse(builder)
#     return matrix
# end

# function flatoffset_sublat(lat, i, no = numorbitals(lat), fo = flatoffsets(lat), o = offsets(lat))
#     s = sublat(lat, i)
#     return (fo[s] + (i - o[s] - 1) * no[s]), s
# end