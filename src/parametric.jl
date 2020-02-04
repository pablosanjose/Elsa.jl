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
    parametric
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
            selected = selector(lat, (row, col), dn)
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

push_ptrdata!(ptrdata, ptr, t::Onsite!{Val{false}}, _...) = push!(ptrdata, ptr)
push_ptrdata!(ptrdata, ptr, t::Hopping!{Val{false}}, _...) = push!(ptrdata, ptr)

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
    applymodifier!.(Ref(ph.h), Ref(ph.originalh), ph.modifiers, ph.ptrdata, Ref(values(kw)))
    return ph.h
end

function applymodifier!(h, oh, modifier, ptrdata, kw)
    for (ohar, har, hardata) in zip(oh.harmonics, h.harmonics, ptrdata)
        nz = nonzeros(har.h)
        onz = nonzeros(ohar.h)
        for data in hardata
            _applymodifier!(nz, onz, modifier, data, kw)
        end
    end
    return h
end

_applymodifier!(nz, onz, modifier, ptr::Int, kw) =
    nz[ptr] = modifier.f(onz[ptr]; kw...)
_applymodifier!(nz, onz, modifier, (ptr, r)::Tuple{Int,SVector}, kw) =
    nz[ptr] = modifier.f(onz[ptr], r; kw...)
_applymodifier!(nz, onz, modifier, (ptr, r, dr)::Tuple{Int,SVector,SVector}, kw) =
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