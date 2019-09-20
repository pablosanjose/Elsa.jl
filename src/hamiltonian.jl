#######################################################################
# Hamiltonian
#######################################################################
struct HamiltonianHarmonic{L,Tv,A<:AbstractMatrix{<:SMatrix{N,N,Tv} where N}}
    dn::SVector{L,Int}
    h::A
end

struct Hamiltonian{L,Tv,H<:HamiltonianHarmonic{L,Tv},F<:Union{Missing,Field}}
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

Base.show(io::IO, h::HamiltonianHarmonic{L,Tv,A}) where
    {L,Tv,N,A<:AbstractArray{<:SMatrix{N,N,Tv}}} = print(io,
"HamiltonianHarmonic{$L,$Tv} with dn = $(Tuple(h.dn)) and elements:", h.h)

displaytype(A::Type{<:SparseMatrixCSC}) = "SparseMatrixCSC, sparse"
displaytype(A::Type{<:Array}) = "Matrix, dense"
displaytype(A::Type) = string(A)

Base.show(io::IO, ham::Hamiltonian{L,Tv,H,F}) where
    {L,Tv,N,F,A<:AbstractArray{<:SMatrix{N,N,Tv}},H<:HamiltonianHarmonic{L,Tv,A}} = print(io,
"Hamiltonian{$L,$Tv} : $(L)D Hamiltonian of element type SMatrix{$N,$N,$Tv}
  Bloch harmonics  : $(length(ham.harmonics)) ($(displaytype(A)))
  Harmonic size    : $((n -> "$n × $n")(nsites(ham)))
  Onsites          : $(nonsites(ham))
  Hoppings         : $(nhoppings(ham))
  Coordination     : $(nhoppings(ham) / nsites(ham))")

# API #

hamiltonian(lat::Lattice, t::TightbindingModelTerm...; kw...) =
    hamiltonian(lat, TightbindingModel(t); kw...)
hamiltonian(lat::Lattice{E,L,T}, m::TightbindingModel; type::Type = Complex{T}, kw...) where {E,L,T} =
    hamiltonian_sparse(blocktype(lat, type), lat, m; kw...)

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
function hamiltonian_sparse(::Type{M}, lat::Lattice{E,L}, model; field = missing) where {E,L,Tv,M<:SMatrix{D,D,Tv} where D}
    builder = IJVBuilder{M}(lat)
    applyterms!(builder, model.terms...)
    HT = HamiltonianHarmonic{L,Tv,SparseMatrixCSC{M,Int}}
    n = nsites(lat)
    harmonics = HT[HT(e.dn, sparse(e.i, e.j, e.v, n, n, (x, xc) -> 0.5 * (x + xc)))
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
            r = lat.unitcell.sites[r]
            vs = orbsized(term(r), lat.unitcell.orbitals[s])
            v = pad(vs, M)
            term.forcehermitian ? push!(ijv, (i, i, v)) : push!(ijv, (i, i, 0.5 * (v + v')))
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
                    push!(ijv, (i, j, v))
                    addadjoint && push!(ijvc, (j, i, v'))
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