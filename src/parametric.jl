#######################################################################
# ParametricHamiltonian
#######################################################################
struct ParametricHamiltonian{N,M<:NTuple{N,ElementModifier},P<:NTuple{N,Any},H<:Hamiltonian}
    originalh::H
    h::H
    modifiers::M  # N modifiers
    ptrdata::P    # P is an NTuple{N,Vector{Vector{ptrdata}}}, one per harmonic
end               # ptrdata may be a nzval ptr, a (ptr,r) or a (ptr, r, dr)

function Base.show(io::IO, ::MIME"text/plain", pham::ParametricHamiltonian{N}) where {N}
    i = get(io, :indent, "")
    print(io, i, "Parametric")
    show(io, pham.h)
    print(io, i, "\n", "$i  Param Modifiers  : $N")
end

"""
    parametric(h::Hamiltonian, modifiers::ElementModifier...)

Builds a `ParametricHamiltonian` that can be used to efficiently apply `modifiers` to `h`.
`modifiers` can be any number of `onsite!(f;...)` and `hopping!(f; ...)` transformations,
each with a set of parameters given as keyword arguments of functions `f`. The resulting
`ph::ParamtricHamiltonian` can be used to produced the modified Hamiltonian simply by
calling it with those same parameters as keyword arguments.

Note that for sparse `h`, `parametric` only modifies existing onsites and hoppings in `h`,
so be sure to add zero onsites and/or hoppings to `h` if they are originally not present but
you need to apply modifiers to them.

    h |> parametric(modifiers::ElementModifier...)

Function form of `parametric`, equivalent to `parametric(h, modifiers...)`.

# Examples
```
julia> ph = LatticePresets.honeycomb() |> hamiltonian(onsite(0) + hopping(1, range = 1/√3)) |> unitcell(10) |> parametric(onsite!((o; μ) -> o - μ))
ParametricHamiltonian{<:Lattice} : Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 5 (SparseMatrixCSC, sparse)
  Harmonic size    : 200 × 200
  Orbitals         : ((:a,), (:a,))
  Element type     : scalar (Complex{Float64})
  Onsites          : 200
  Hoppings         : 600
  Coordination     : 3.0
  Param Modifiers  : 1

julia> ph(; μ = 2)
Hamiltonian{<:Lattice} : Hamiltonian on a 2D Lattice in 2D space
  Bloch harmonics  : 5 (SparseMatrixCSC, sparse)
  Harmonic size    : 200 × 200
  Orbitals         : ((:a,), (:a,))
  Element type     : scalar (Complex{Float64})
  Onsites          : 200
  Hoppings         : 600
  Coordination     : 3.0

# See also
    `onsite!`, `hopping!`
```
"""
parametric(h::Hamiltonian, ts::ElementModifier...) =
    ParametricHamiltonian(h, copy(h), ts, parametric_pointers.(Ref(h), ts))
parametric(ts::ElementModifier...) = h -> parametric(h, ts...)

function parametric_pointers(h::Hamiltonian{LA,L,M,<:AbstractSparseMatrix}, t::ElementModifier) where {LA,L,M}
    harmonic_ptrdata = empty_ptrdata(h, t)
    lat = h.lattice
    selector = resolve(t.selector, lat)
    for (har, ptrdata) in zip(h.harmonics, harmonic_ptrdata)
        matrix = har.h
        dn = har.dn
        rows = rowvals(matrix)
        for col in 1:size(matrix, 2), ptr in nzrange(matrix, col)
            row = rows[ptr]
            selected = selector(lat, (row, col), (dn, zero(dn)))
            selected && push_ptrdata!(ptrdata, ptr, t, lat, (row, col))
        end
    end
    return harmonic_ptrdata
end

# needspositions = false, one vector of nzval ptr per harmonic
empty_ptrdata(h, t::Onsite!{Val{false}})  = [Int[] for _ in h.harmonics]
empty_ptrdata(h, t::Hopping!{Val{false}}) = [Int[] for _ in h.harmonics]
# needspositions = true, one vector of (ptr, r, dr) per harmonic
function empty_ptrdata(h, t::Onsite!{Val{true}})
    S = positiontype(h.lattice)
    return [Tuple{Int,S}[] for _ in h.harmonics]
end
function empty_ptrdata(h, t::Hopping!{Val{true}})
    S = positiontype(h.lattice)
    return [Tuple{Int,S,S}[] for _ in h.harmonics]
end

push_ptrdata!(ptrdata, ptr, t::ElementModifier{Val{false}}, _...) = push!(ptrdata, ptr)

function push_ptrdata!(ptrdata, ptr, t::Onsite!{Val{true}}, lat, (row, col))
    r = sites(lat)[col]
    push!(ptrdata, (ptr, r))
end

function push_ptrdata!(ptrdata, ptr, t::Hopping!{Val{true}}, lat, (row, col))
    r, dr = _rdr(sites(lat)[col], sites(lat)[row])
    push!(ptrdata, (ptr, r, dr))
end

function (ph::ParametricHamiltonian)(; kw...)
    checkconsistency(ph, false) # only weak check for performance
    applymodifier_ptrdata!.(Ref(ph.h), Ref(ph.originalh), ph.modifiers, ph.ptrdata, Ref(values(kw)))
    return ph.h
end

function applymodifier_ptrdata!(h, oh, modifier, ptrdata, kw)
    for (ohar, har, hardata) in zip(oh.harmonics, h.harmonics, ptrdata)
        nz = nonzeros(har.h)
        onz = nonzeros(ohar.h)
        for data in hardata
            _applymodifier_ptrdata!(nz, onz, modifier, data, kw)
        end
    end
    return h
end

_applymodifier_ptrdata!(nz, onz, modifier, ptr::Int, kw) =
    nz[ptr] = modifier.f(onz[ptr]; kw...)
_applymodifier_ptrdata!(nz, onz, modifier, (ptr, r)::Tuple{Int,SVector}, kw) =
    nz[ptr] = modifier.f(onz[ptr], r; kw...)
_applymodifier_ptrdata!(nz, onz, modifier, (ptr, r, dr)::Tuple{Int,SVector,SVector}, kw) =
    nz[ptr] = modifier.f(onz[ptr], r, dr; kw...)

function checkconsistency(ph::ParametricHamiltonian, fullcheck = true)
    isconsistent = true
    length(ph.originalh.harmonics) == length(ph.h.harmonics) || (isconsitent = false)
    if fullcheck && isconsistent
        for (ohar, har) in zip(ph.originalh.harmonics, ph.h.harmonics)
            length(nonzeros(har.h)) == length(nonzeros(ohar.h)) || (isconsistent = false; break)
            rowvals(har.h) == rowvals(ohar.h) || (isconsistent = false; break)
            getcolptr(har.h) == getcolptr(ohar.h) || (isconsistent = false; break)
        end
    end
    isconsistent ||
        throw(error("ParametricHamiltonian is not internally consistent, it may have been modified after creation"))
    return nothing
end