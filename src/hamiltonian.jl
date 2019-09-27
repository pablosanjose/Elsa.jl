#######################################################################
# Hamiltonian
#######################################################################
struct HamiltonianHarmonic{L,M,A<:AbstractMatrix{M}}
    dn::SVector{L,Int}
    h::A
end

struct Hamiltonian{LA<:AbstractLattice,L,M,H<:HamiltonianHarmonic{L,M},F<:Union{Missing,Field},A<:Union{Missing,AbstractMatrix}}
    lattice::LA
    harmonics::Vector{H}
    field::F
    matrix::A
end

function Hamiltonian(lat, hs::Vector{H}, field, n::Int, m::Int) where {L,M,H<:HamiltonianHarmonic{L,M}}
    isempty(hs) && push!(hs, H(zero(SVector{L,Int}), empty_sparse(M, n, m)))
    sort!(hs, by = h -> abs.(h.dn))
    return Hamiltonian(lat, hs, field)
end
Hamiltonian(lat::Superlattice, hs, field) = Hamiltonian(lat, hs, field, missing)
Hamiltonian(lat::Lattice, hs, field) = Hamiltonian(lat, hs, field, optimized_h0(hs))

function Base.show(io::IO, ham::Hamiltonian)
    i = get(io, :indent, "")
    print(io, i, summary(ham), "\n",
"$i  Bloch harmonics  : $(length(ham.harmonics)) ($(displaymatrixtype(ham)))
$i  Harmonic size    : $((n -> "$n × $n")(nsites(ham)))
$i  Elements         : $(displayelements(ham))
$i  Onsites          : $(nonsites(ham))
$i  Hoppings         : $(nhoppings(ham))
$i  Coordination     : $(nhoppings(ham) / nsites(ham))")
end

Base.show(io::IO, h::HamiltonianHarmonic{L,M}) where {L,M} = print(io,
"HamiltonianHarmonic{$L,$(eltype(M))} with dn = $(Tuple(h.dn)) and elements:", h.h)

Base.summary(::Hamiltonian{LA}) where {E,L,LA<:Lattice{E,L}} =
    "Hamiltonian{<:Lattice} : $(L)D Hamiltonian on a $(L)D Lattice in $(E)D space"

Base.summary(::Hamiltonian{LA}) where {E,L,T,L´,LA<:Superlattice{E,L,T,L´}} =
    "Hamiltonian{<:Superlattice} : $(L)D Hamiltonian on a $(L´)D Superlattice in $(E)D space"

matrixtype(::Hamiltonian{LA,L,M,H}) where {LA,L,M,A,H<:HamiltonianHarmonic{L,M,A}} = A
displaymatrixtype(h::Hamiltonian) = displaymatrixtype(matrixtype(h))
displaymatrixtype(::Type{<:SparseMatrixCSC}) = "SparseMatrixCSC, sparse"
displaymatrixtype(::Type{<:Array}) = "Matrix, dense"
displaymatrixtype(A::Type{<:AbstractArray}) = string(A)
displayelements(h::Hamiltonian) = displayelements(blocktype(h))
displayelements(::Type{<:SMatrix{N,N}}) where {N} = "$N × $N blocks"
displayelements(::Type{<:Number}) = "scalars"

# work matrix to avoid reallocations when summing harmonics
function optimized_h0(hs::Vector{HamiltonianHarmonic{L,M,A}}) where {L,M,A<:SparseMatrixCSC}
    Tv = eltype(M)
    h0 = first(hs)
    n, m = size(h0.h)
    iszero(h0.dn) || throw(ArgumentError("First Hamiltonian harmonic is not the fundamental"))
    nh = length(hs)
    builder = SparseMatrixBuilder{M}(n, m)
    for col in 1:m
        for i in eachindex(hs)
            h = hs[i].h
            for p in nzrange(h, col)
                v = i == 1 ? nonzeros(h)[p] : zero(M)
                row = rowvals(h)[p]
                pushtocolumn!(builder, row, v, false) # skips repeated rows
            end
        end
        finalisecolumn!(builder)
    end
    ho = sparse(builder)
    return ho
end

# Internal API #

blocktype(h::Hamiltonian{LA,L,M}) where {LA,L,M} = M

function nhoppings(ham::Hamiltonian)
    count = 0
    for h in ham.harmonics
        count += iszero(h.dn) ? (_nnz(h.h) - nnzdiag(h.h)) : _nnz(h.h)
    end
    return count
end

function nonsites(ham::Hamiltonian)
    count = 0
    for h in ham.harmonics
        iszero(h.dn) && (count += nnzdiag(h.h))
    end
    return count
end

_nnz(h::SparseMatrixCSC) = nnz(h)
_nnz(h::Matrix) = length(h)

nsites(h::Hamiltonian) = isempty(h.harmonics) ? 0 : size(first(h.harmonics).h, 1)

# API #
"""
    hamiltonian(lat::Lattice{E,L,T}, models...; type = Complex{T}, field = missing)

Create a `Hamiltonian` by additively applying `models` to the lattice `lat`` (see `hopping`
and `onsite` for details on building tightbinding models).

The elements of the Hamiltonian are of type `type`, or `SMatrix{N,N,type}`, depending on the
orbitals in `lat`. The `model` must match said orbitals.

Advanced use: if a `field = f(r,dr,h)` function is given, it will affect how the hamiltonian
operates on a `State`, and in combination with `superlattice` allows to include matrix-free
perturbations (e.g. disorder, gauge fields) when acting on a `State` (see documentation for
details).

    hamiltonian(lat, func::Function, models...; kw...)

For a function of the form `func(;kw...)::AbstractTightbindingModel`, this produces a
`h::ParametricHamiltonian` that efficiently generates a `Hamiltonian` when calling it as in
`h(;kw...)`.

    hamiltonian(superlat, h::Hamiltonian)

Builds a `Hamiltonian` by applying `h` to superlattice `superlat`. The seed `h` should be
compatible with `superlattice`, i.e. `iscompatible(superlattice, h) == true` (see also
`superlattice`).

    lat |> hamiltonian(h::Hamiltonian)
    lat |> hamiltonian([func, ] models...)

Functional form of `hamiltonian`, equivalent to `hamiltonian(lat, args...)`

# Examples
```jldoctest
julia> hamiltonian(LatticePresets.honeycomb(), hopping(1, range = 1/√3))
Hamiltonian{2,Complex{Float64}} : 2D Hamiltonian (scalar elements)
  Bloch harmonics  : 5 (SparseMatrixCSC, sparse)
  Harmonic size    : 2 × 2
  Onsites          : 0
  Hoppings         : 6
  Coordination     : 3.0
```
"""
hamiltonian(lat::AbstractLattice, t::AbstractTightbindingModel...; kw...) =
    hamiltonian(lat, TightbindingModel(t...); kw...)
hamiltonian(lat::AbstractLattice, m::TightbindingModel; type::Type = Complex{numbertype(lat)}, kw...) =
    hamiltonian_sparse(blocktype(lat, type), lat, m; kw...)

hamiltonian(lat::AbstractLattice, f::Function, ts::AbstractTightbindingModel...;
            type::Type = Complex{numbertype(lat)}, kw...) =
    parametric_hamiltonian(blocktype(lat, type), lat, f, TightbindingModel(ts...); kw...)

hamiltonian(t::AbstractTightbindingModel...; kw...) =
    z -> hamiltonian(z, t...; kw...)
hamiltonian(f::Function, t::AbstractTightbindingModel...; kw...) =
    z -> hamiltonian(z, f, t...; kw...)
hamiltonian(h::Hamiltonian) =
    z -> hamiltonian(z, h)

(h::Hamiltonian)(phases...) = bloch(h, phases...)

bloch(h::Hamiltonian{<:Lattice}, phases...) = copy(bloch!(h, phases...))

bloch!(h::Hamiltonian, phases::Vararg{Number,L}) where {L} = bloch!(h, SVector{L}(phases))
bloch!(h::Hamiltonian, phases::NTuple{L,Number}) where {L} = bloch!(h, SVector{L}(phases))
function bloch!(h::Hamiltonian{<:Lattice,L,M,H}, phases::SVector{L}) where {L,M,A<:SparseMatrixCSC,H<:HamiltonianHarmonic{L,M,A}}
    h0 = first(h.harmonics).h
    matrix = h.matrix
    if length(h0.nzval) == length(matrix.nzval) # rewrite matrix from previous calls
        copy!(matrix.nzval, h0.nzval)
    else # first call, align first harmonic h0 with optimized matrix
        copy!(h0.colptr, matrix.colptr)
        copy!(h0.rowval, matrix.rowval)
        copy!(h0.nzval, matrix.nzval)
    end
    for ns in 2:length(h.harmonics)
        hh = h.harmonics[ns]
        ephi = cis(phases' * hh.dn)
        muladd_optsparse(matrix, ephi, hh.h)
    end
    return matrix
end

function muladd_optsparse(matrix, ephi, h)
    for col in 1:size(h,2)
        range = nzrange(h, col)
        for ptr in range
            row = h.rowval[ptr]
            matrix[row, col] = ephi * h.nzval[ptr]
        end
    end
    return nothing
end

# function _isnz(matrix, row, col)
#     rows = rowvals(matrix)
#     for p in nzrange(matrix, col)
#         rows[p] == row && return true
#     end
#     return false
# end
# iscompatible(lat::AbstractLattice{E,L}, h::Hamiltonian{L,M}) where {E,L,M} =
#     blocktype(lat, eltype(M)) == blocktype(h) && nsites(h) == nsites(lat)
# iscompatible(lat::AbstractLattice{E,L}, h::Hamiltonian{L2,M}) where {E,L,L2,M} =
#     false

Base.Matrix(h::Hamiltonian) = Hamiltonian(h.lattice, Matrix.(h.harmonics), h.field, h.matrix)
Base.Matrix(h::HamiltonianHarmonic) = HamiltonianHarmonic(h.dn, Matrix(h.h))

Base.copy(h::Hamiltonian) = Hamiltonian(h.lattice, copy.(h.harmonics), h.field)
Base.copy(h::HamiltonianHarmonic) = HamiltonianHarmonic(h.dn, copy(h.h))

Base.size(h::Hamiltonian, n) = size(first(h.harmonics).h, n)
Base.size(h::Hamiltonian) = size(first(h.harmonics).h)
Base.size(h::HamiltonianHarmonic, n) = size(h.h, n)
Base.size(h::HamiltonianHarmonic) = size(h.h)

#######################################################################
# auxiliary types
#######################################################################
struct IJV{L,M}
    dn::SVector{L,Int}
    i::Vector{Int}
    j::Vector{Int}
    v::Vector{M}
end

struct IJVBuilder{L,M,E,T,LA<:AbstractLattice{E,L,T}}
    lat::LA
    ijvs::Vector{IJV{L,M}}
    kdtrees::Vector{KDTree{SVector{E,T},Euclidean,T}}
end

IJV{L,M}(dn::SVector{L} = zero(SVector{L,Int})) where {L,M} =
    IJV(dn, Int[], Int[], M[])

function IJVBuilder{M}(lat::AbstractLattice{E,L,T}) where {E,L,T,M}
    ijvs = IJV{L,M}[]
    kdtrees = Vector{KDTree{SVector{E,T},Euclidean,T}}(undef, nsublats(lat))
    return IJVBuilder(lat, ijvs, kdtrees)
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
function hamiltonian_sparse(::Type{M}, lat::AbstractLattice{E,L}, model; field = missing) where {E,L,M}
    builder = IJVBuilder{M}(lat)
    applyterms!(builder, terms(model)...)
    HT = HamiltonianHarmonic{L,M,SparseMatrixCSC{M,Int}}
    n = nsites(lat)
    harmonics = HT[HT(e.dn, sparse(e.i, e.j, e.v, n, n)) for e in builder.ijvs if !isempty(e)]
    return Hamiltonian(lat, harmonics, Field(field, lat), n, n)
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
            vs = orbsized(term(r), lat.unitcell.orbitals[s])
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
                    vs = orbsized(term(r, dr), lat.unitcell.orbitals[s1], lat.unitcell.orbitals[s2])
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
# unitcell/supercell
#######################################################################

function supercell(ham::Hamiltonian, args...; kw...)
    slat = supercell(ham.lattice, args...; kw...)
    return Hamiltonian(slat, ham.harmonics, ham.field)
end

function unitcell(ham::Hamiltonian{<:Lattice}, args...; kw...)
    sham = supercell(ham, args...; kw...)
    return unitcell(sham)
end

function unitcell(ham::Hamiltonian{LA,L,Tv}) where {E,L,T,L´,Tv,LA<:Superlattice{E,L,T,L´}}
    lat = ham.lattice
    mapping = similar(lat.supercell.cellmask, Int) # store supersite indices newi
    mapping .= 0
    foreach_supersite((s, oldi, olddn, newi) -> mapping[oldi, Tuple(olddn)...] = newi, lat)
    dim = nsites(lat.supercell)
    B = blocktype(ham)
    harmonic_builders = HamiltonianHarmonic{L´,Tv,SparseMatrixBuilder{B}}[]
    pinvint = pinvmultiple(lat.supercell.matrix)
    foreach_supersite(lat) do s, source_i, source_dn, newcol
        for oldh in ham.harmonics
            rows = rowvals(oldh.h)
            vals = nonzeros(oldh.h)
            target_dn = source_dn + oldh.dn
            super_dn = new_dn(target_dn, pinvint)
            wrapped_dn = wrap_dn(target_dn, super_dn, lat.supercell.matrix)
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
    field = ham.field
    unitlat = unitcell(lat)
    return Hamiltonian(unitlat, harmonics, field)
end

function get_or_push!(hs::Vector{HamiltonianHarmonic{L,Tv,SparseMatrixBuilder{B}}}, dn, dim) where {L,Tv,B}
    for h in hs
        h.dn == dn && return h
    end
    newh = HamiltonianHarmonic(dn, SparseMatrixBuilder{B}(dim, dim))
    push!(hs, newh)
    return newh
end

# hamiltonian(lat::Lattice{E,L,T,S}, ham::Hamiltonian{L2,Tv}) where {E,L,T,S,L2,Tv} =
#     throw(DimensionMismatch("Lattice dimensions $L does not match the Hamiltonian's $L2"))

#######################################################################
# parametric hamiltonian
#######################################################################
struct ParametricHamiltonian{H,F,E,T}
    base::H
    hamiltonian::H
    pointers::Vector{Vector{Tuple{Int,SVector{E,T},SVector{E,T}}}} # val pointers to modify
    f::F                                                           # by f on each harmonic
end

Base.eltype(::ParametricHamiltonian{H}) where {L,M,H<:Hamiltonian{L,M}} = M

Base.show(io::IO, pham::ParametricHamiltonian) = print(io, "Parametric ", pham.hamiltonian)

function parametric_hamiltonian(::Type{M}, lat::Lattice{E,L,T}, f::F, model; field = missing) where {M,E,L,T,F}
    builder = IJVBuilder{M}(lat)
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
    base_h = Hamiltonian(base_harmonics, missing, n, n)
    h = Hamiltonian(harmonics, Field(field, lat), n, n)
    return ParametricHamiltonian(base_h, h, pointers, f)
end

function getpointers(h::SparseMatrixCSC, ijv, eloffset, lat::Lattice{E,L,T}) where {E,L,T}
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

function (ph::ParametricHamiltonian)(;kw...)
    isempty(kw) && return ph.hamiltonian
    model = ph.f(;kw...)
    initialize!(ph)
    foreach(term -> applyterm!(ph, term), terms(model))
    return ph.hamiltonian
end

function initialize!(ph::ParametricHamiltonian)
    for (bh, h, prdrs) in zip(ph.base.harmonics, ph.hamiltonian.harmonics, ph.pointers)
        vals = nonzeros(h.h)
        vals_base = nonzeros(bh.h)
        for (p,_,_) in prdrs
            vals[p] = vals_base[p]
        end
    end
    return nothing
end

function applyterm!(ph::ParametricHamiltonian{H}, term::TightbindingModelTerm) where {L,M,H<:Hamiltonian{L,M}}
    for (h, prdrs) in zip(ph.hamiltonian.harmonics, ph.pointers)
        vals = nonzeros(h.h)
        for (p, r, dr) in prdrs
            v = term(r, dr) # should perhaps be v = orbsized(term(r, dr), orb1, orb2)
            vals[p] += padtotype(v, M)
        end
    end
    return nothing
end

#######################################################################
# bloch
#######################################################################
# struct BlochHamiltonian{L,M,A,H<:HamiltonianHarmonic{L,M,A}}
#     harmonics::Vector{H}
#     matrix::A
# end

# Base.show(io::IO, h::BlochHamiltonian{L,M}) where {L,M} = print(io,
# "BlochHamiltonian{$L,$(eltype(M))} of a $(L)D Hamiltonian with $(length(h.harmonics)) harmonics")

# function BlochHamiltonian(h::Hamiltonian)
#     hh0 = first(h.harmonics)
#     iszero(hh0.dn) || throw(ArgumentError("First Hamiltonian harmonic is not the fundamental"))
#     return BlochHamiltonian(h.harmonics, copy(hh0.h))
# end

# (h::BlochHamiltonian)(phases::Vararg{Number,L}) where {L} = h(SVector{L}(phases))
# (h::BlochHamiltonian)(phases::NTuple{L,Number}) where {L} = h(SVector{L}(phases))
# function (h::BlochHamiltonian{L,M,<:SparseMatrixCSC})(phases::SVector{L}) where {L,M}
#     h0 = first(h.harmonics).h
#     matrix = h.matrix
#     copy!(matrix.nzval, h0.nzval)
#     for ns in 2:length(h.harmonics)
#         hh = h.harmonics[ns]
#         ephi = cis(phases' * hh.dn)
#         muladd_optsparse(matrix, ephi, hh.h)
#     end
#     return matrix
# end

# bloch(h::Hamiltonian) = BlochHamiltonian(optimize!(copy(h)))

# function muladd_optsparse(matrix, ephi, h)
#     for col in 1:size(h,2)
#         range = nzrange(h, col)
#         for ptr in range
#             row = h.rowval[ptr]
#             matrix[row, col] = ephi * h.nzval[ptr]
#         end
#     end
#     return nothing
# end

# # # avoid reallocations when summing harmonics
# # function optimize!(h::Hamiltonian{L,M,H}) where {L,M,A<:SparseMatrixCSC,H<:HamiltonianHarmonic{L,M,A}}
# #     Tv = eltype(M)
# #     small = eps(real(Tv))
# #     hh0 = first(h.harmonics)
# #     iszero(hh0.dn) || throw(ArgumentError("First Hamiltonian harmonic is not the fundamental"))
# #     nh = length(h.harmonics)
# #     for i in 2:nh
# #         hh0.h .+= small * h.harmonics[i].h
# #     end
# #     return h
# # end