#######################################################################
# Hamiltonian
#######################################################################
struct HamiltonianHarmonic{L,Tv,A<:AbstractMatrix{<:SMatrix{N,N,Tv} where N}}
    dn::SVector{L,Int}
    h::A
end

struct Hamiltonian{L,Tv,H<:HamiltonianHarmonic{L,Tv},D<:Domain{L}}
    harmonics::Vector{H}
    domain::D
end

struct HamiltonianBuilder{L,Tv,H<:Hamiltonian{L,Tv},F<:Function,P<:NamedTuple}
    h::H
    field::F
    parameters::P
end

blocktype(::Type{Tv}, lat::Lattice) where {Tv} = 
    blocktype(SMatrix{1,1,Tv,1}, lat.sublats...)
blocktype(::Type{S}, s::Sublat{E,T,D}, ss...) where {N,Tv,E,T,D,S<:SMatrix{N,N,Tv}} = 
    (M = max(N,D); blocktype(SMatrix{M,M,Tv,M^2}, ss...))
blocktype(t) = t

Base.size(h::Hamiltonian) = (n = size(h.domain.bitmask)[end]; (n, n))


Base.show(io::IO, ham::Hamiltonian{L,Tv,H}) where {L,Tv,N,
    H<:HamiltonianHarmonic{L,Tv,<:AbstractArray{<:SMatrix{N,N,Tv}}}} = print(io, 
"Hamiltonian{$L,$Tv} : $(L)D Hamiltonian of element type SMatrix{$N,$N,$Tv}
  Bloch harmonics  : $(length(ham.harmonics))
  Harmonic size    : $(size(ham))")
#   Bravais vectors     : $(displayvectors(sys.lattice.bravais))
#   Sublattice names    : $((sublatnames(sys.lattice)... ,))
#   Sublattice orbitals : $((norbitals(sys)... ,))
#   Total sites         : $(nsites(sys)) [$T]
#   Total hoppings      : $(nlinks(sys)) [$Tv]
#   Coordination        : $(coordination(sys))")

# API #

hamiltonian(lat::Lattice, t::TightbindingModelTerm...; kw...) = 
    hamiltonian(lat, TightbindingModel(t))
hamiltonian(lat::Lattice{E,L,T}, m::TightbindingModel; htype::Type = Complex{T}) where {E,L,T} = 
    sparse_hamiltonian(blocktype(htype, lat), lat, m.terms...)

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

function Base.getindex(b::IJVBuilder{L,M}, dn::SVector{L,Int}) where {L,M}
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
# sparse_hamiltonian
#######################################################################
function sparse_hamiltonian(::Type{M}, lat::Lattice{E,L}, terms...) where {E,L,Tv,M<:SMatrix{D,D,Tv} where D}
    builder = IJVBuilder{M}(lat)
    applyterms!(builder, terms...)
    HT = HamiltonianHarmonic{L,Tv,SparseMatrixCSC{M,Int}}
    n = nsites(lat)
    harmonics = HT[HT(e.dn, sparse(e.i, e.j, e.v, n, n))  for e in builder.ijvs 
        if !isempty(e)]
    return Hamiltonian(harmonics, lat.domain)
end

applyterms!(builder, terms...) = foreach(term -> applyterm!(builder, term), terms)

applyterm!(builder, term::OnsiteTerm) =
    foreach(s -> applyterm!(builder, term, builder.lat.sublats[s], s), sublats(term, builder.lat))

# Function barrier for type-stable sublat
function applyterm!(builder::IJVBuilder{L,M}, term::OnsiteTerm, 
                    sublat::Sublat{E,T,D}, s) where {L,E,T,D,Tv,M<:SMatrix{Dp,Dp,Tv} where Dp} 
    dn0 = zero(SVector{L,Int})
    ijv = builder[dn0]
    offset = builder.lat.offsets[s]
    for (n, r) in enumerate(sublat.sites)
        i = offset + n
        v = pad(SMatrix{D,D,Tv}(term(r)), M)
        push!(ijv, (i, i, v))
    end
    return nothing
end

function applyterm!(builder, term::HoppingTerm)
    checkinfinite(term)
    foreach(sublats(term, builder.lat)) do ss 
        applyterm!(builder, term, ss,
                   builder.lat.sublats[first(ss)], builder.lat.sublats[last(ss)])
    end
    return nothing
end

# Function barrier for type-stable sublat1 and sublat2
function applyterm!(builder::IJVBuilder{L,M}, term::HoppingTerm, (s1, s2),
                    sublat1::Sublat{E,T,D1}, sublat2::Sublat{E,T,D2}) where {L,E,T,D1,D2,Tv,
                                                                 M<:SMatrix{D,D,Tv} where D}
    dns = dniter(term.dns, Val(L))
    for dn in dns
        foundlink = false
        addadjoint = needsadjoint(term, (s1, s2), dn)
        ijv = builder[dn]
        addadjoint && (ijvc = builder[negative(dn)])
        for (j, site) in enumerate(sublat2.sites)
            rsource = site - builder.lat.bravais.matrix * dn
            itargets = targets(builder, term.range, rsource, s1)
            for i in itargets
                isselfhopping((s1, s2), (i, j), dn) && continue
                foundlink = true
                rtarget = sublat1.sites[i]
                r, dr = _rdr(rsource, rtarget)
                v = pad(SMatrix{D1,D2,Tv}(term(r, dr)), M)
                push!(ijv, (i, j, v))
                addadjoint && push!(ijvc, (j, i, v'))
            end
        end
        foundlink && acceptcell!(dns, dn)
    end
    return nothing
end

# If dn are specified in model term (not missing), iterate over them. Otherwise do a search.
dniter(dns::Missing, ::Val{L}) where {L} = BoxIterator(zero(SVector{L,Int}))
dniter(dns, ::Val) = dns

function targets(builder, range::Real, rsource, s1)
    isassigned(builder.kdtrees, s1) || (builder.kdtrees[s1] = KDTree(builder.lat.sublats[s1].sites))
    return inrange(builder.kdtrees[s1], rsource, range)
end

targets(builder, range::Missing, rsource, s1) = eachindex(builder.lat.sublats[s1].sites)

checkinfinite(term) = term.dns === missing && (term.range === missing || !isfinite(term.range)) && 
    throw(ErrorException("Tried to implement an infinite-range hopping on an unbounded lattice"))

isselfhopping((s1, s2), (i, j), dn) = i == j && s1 == s2 && iszero(dn)

isvalidlink(term, (s1, s2), dn) = needsadjoint(term, (s1, s2), dn) || (iszero(dn) && s1 == s2)

needsadjoint(h::HoppingTerm{F,Missing,Missing}, (s1, s2), dn) where {F} = 
    h.forcehermitian && (s1 != s2 || ispositive(dn))
needsadjoint(h::HoppingTerm{F,S,Missing}, (s1, s2), dn) where {F,S} = 
    h.forcehermitian && (s1 != s2 || ispositive(dn))
needsadjoint(h::HoppingTerm{F,Missing}, (s1, s2), dn) where {F} = 
    h.forcehermitian && (s1 != s2 || !iszero(dn))
needsadjoint(h::HoppingTerm, (s1, s2), dn) = 
    h.forcehermitian && (s1 != s2 || !iszero(dn))

function ispositive(dn)
    result = false
    for i in dn
        i == 0 || (result = i > 0; break)
    end
    return result
end