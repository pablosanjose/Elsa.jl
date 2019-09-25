#######################################################################
# Hamiltonian
#######################################################################
struct HamiltonianHarmonic{L,M,A<:AbstractMatrix{M}}
    dn::SVector{L,Int}
    h::A
end

struct Hamiltonian{L,M,H<:HamiltonianHarmonic{L,M},F<:Union{Missing,Field}}
    harmonics::Vector{H}
    field::F
end

nsites(h::Hamiltonian) = isempty(h.harmonics) ? 0 : size(first(h.harmonics).h, 1)

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

Base.Matrix(h::Hamiltonian) = Hamiltonian(Matrix.(h.harmonics), h.field, h.lattice)
Base.Matrix(h::HamiltonianHarmonic) = HamiltonianHarmonic(h.dn, Matrix(h.h))

blocktype(h::Hamiltonian{L,M}) where {L,M} = M

iscompatible(lat::Lattice{E,L}, h::Hamiltonian{L,M}) where {E,L,M} =
    blocktype(lat, eltype(M)) == blocktype(h) && nsites(h) == nsites(lat)
iscompatible(lat::Lattice{E,L}, h::Hamiltonian{L2,M}) where {E,L,L2,M} =
    false

Base.show(io::IO, h::HamiltonianHarmonic{L,M}) where {L,M} = print(io,
"HamiltonianHarmonic{$L,$(eltype(M))} with dn = $(Tuple(h.dn)) and elements:", h.h)

displaytype(A::Type{<:SparseMatrixCSC}) = "SparseMatrixCSC, sparse"
displaytype(A::Type{<:Array}) = "Matrix, dense"
displaytype(A::Type) = string(A)

displayblock(::Type{<:SMatrix{N,N}}) where {N} = "$N × $N block elements"
displayblock(::Type{<:Number}) = "scalar elements"

function Base.show(io::IO, ham::Hamiltonian{L,M,H}) where {L,M,A,H<:HamiltonianHarmonic{L,M,A}}
    i = get(io, :indent, "")
    print(io,
"$(i)Hamiltonian{$L,$(eltype(M))} : $(L)D Hamiltonian ($(displayblock(M)))
$(i)  Bloch harmonics  : $(length(ham.harmonics)) ($(displaytype(A)))
$(i)  Harmonic size    : $((n -> "$n × $n")(nsites(ham)))
$(i)  Onsites          : $(nonsites(ham))
$(i)  Hoppings         : $(nhoppings(ham))
$(i)  Coordination     : $(nhoppings(ham) / nsites(ham))")
end

# API #

hamiltonian(lat::Lattice, t::AbstractTightbindingModel...; kw...) =
    hamiltonian(lat, TightbindingModel(t...); kw...)
hamiltonian(lat::Lattice, m::TightbindingModel; type::Type = Complex{sitetype(lat)}, kw...) =
    hamiltonian_sparse(blocktype(lat, type), lat, m; kw...)

hamiltonian(t::AbstractTightbindingModel...; kw...) = z -> hamiltonian(z, t...; kw...)
hamiltonian(h::Hamiltonian) = z -> hamiltonian(z, h)

#######################################################################
# auxiliary types
#######################################################################
struct IJV{L,M}
    dn::SVector{L,Int}
    i::Vector{Int}
    j::Vector{Int}
    v::Vector{M}
end

struct IJVBuilder{L,M,E,T,LA<:Lattice{E,L,T}}
    lat::LA
    ijvs::Vector{IJV{L,M}}
    kdtrees::Vector{KDTree{SVector{E,T},Euclidean,T}}
end

IJV{L,M}(dn::SVector{L} = zero(SVector{L,Int})) where {L,M} =
    IJV(dn, Int[], Int[], M[])

function IJVBuilder{M}(lat::Lattice{E,L,T}) where {E,L,T,M}
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
function hamiltonian_sparse(::Type{M}, lat::Lattice{E,L}, model; field = missing) where {E,L,M}
    builder = IJVBuilder{M}(lat)
    applyterms!(builder, model.terms...)
    HT = HamiltonianHarmonic{L,M,SparseMatrixCSC{M,Int}}
    n = nsites(lat)
    harmonics = HT[HT(e.dn, sparse(e.i, e.j, e.v))#, n, n, (x, xc) -> 0.5 * (x + xc)))
                   for e in builder.ijvs if !isempty(e)]
    return Hamiltonian(harmonics, Field(field, lat))
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
            v = pad(vs, M)
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
                    v = pad(vs, M)
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

# If all sublats are scanned, avoid doubling hoppings when adding adjoint
redundancyfactor(dn, ss, term) =
    isnotredundant(dn, term) || isnotredundant(ss, term) ? 1.0 : 0.5
# (i,j,dn) and (j,i,-dn) will not both be added if any of the following is true
isnotredundant(dn::SVector, term) = term.dns !== missing && !iszero(dn)
isnotredundant((s1, s2)::Tuple{Int,Int}, term) = term.sublats !== missing && s1 != s2

#######################################################################
# hamiltonian(lattice, hamiltonian)
#######################################################################
function hamiltonian(lat::Lattice{E,L,T,S}, ham::Hamiltonian{L,Tv}) where {L,Tv,E,T,L´,S<:Supercell{L,L´}}
    iscompatible(lat, ham) || throw(ArgumentError("Lattice and Hamiltonian are incompatible"))
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
                # wrapped_dn could exit bounding box along non-periodic direction
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
    return Hamiltonian(harmonics, field)
end

function get_or_push!(hs::Vector{HamiltonianHarmonic{L,Tv,SparseMatrixBuilder{B}}}, dn, dim) where {L,Tv,B}
    for h in hs
        h.dn == dn && return h
    end
    newh = HamiltonianHarmonic(dn, SparseMatrixBuilder{B}(dim, dim))
    push!(hs, newh)
    return newh
end

hamiltonian(lat::Lattice{E,L,T,S}, ham::Hamiltonian{L2,Tv}) where {E,L,T,S,L2,Tv} =
    throw(DimensionMismatch("Lattice dimensions $L does not match the Hamiltonian's $L2"))