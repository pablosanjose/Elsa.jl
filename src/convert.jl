# @inline toNTuples(::Type{T}, t, ts...) where {T} = (toNTuples(T, t)..., toNTuples(T, ts...)...)
# @inline toNTuples(::Type{T}, t::Vector) where {T} = (ntuple(i->T(t[i]), length(t)),)
# @inline toNTuples(::Type{T}, t::SVector{N,T2}) where {N,T,T2} = (convert(NTuple{N,T}, Tuple(t)),)
# @inline toNTuples(::Type{T}, t::NTuple{N,Any}) where {N, T} = (convert(NTuple{N,T}, t),)
# @inline toNTuples(::Type{T}, t::Number) where {T} = (T(t),)

# In v0.7+ the idea is that `convert`` is a shortcut to a "safe" subset of the constructors 
# for a type, that guarantees the resulting type. The constructor is the central machinery
# to instantiate types and convert between instances. (For parametric types, unless overridden, 
# you get an implicit internal constructor without parameters, so no need to define that externally)

convert(::Type{T}, l::T) where T<:Lattice = l
convert(::Type{T}, l::Lattice) where T<:Lattice = T(l)

convert(::Type{T}, l::T) where T<:LatticeOption = l
convert(::Type{T}, l::LatticeOption) where T<:LatticeOption = T(l)

# convert(::Type{Matrix{Slink{T,E}}}, nsublats::Int) where {T,E} =
#     [Slink{T,E}() for i in 1:nsublats, j in 1:nsublats]

# Constructors for conversion

Sublat{T,E}(s::Sublat) where {T,E} = 
    Sublat(s.name, [padright(site, zero(T), Val(E)) for site in s.sites])

Bravais{T,E,L,EL}(b::Bravais) where {T,E,L,EL} = 
    Bravais(padrightbottom(b.matrix, SMatrix{E,L,T,EL}))

function Slink{T,E}(s::Slink) where {T,E}
    rdr = Vector{Tuple{SVector{E,T}, SVector{E,T}}}(undef, length(s.rdr))
    @inbounds for (i, ri) in enumerate(s.rdr)
        rdr[i] = (padright(ri[1], zero(T), Val(E)), padright(ri[2], zero(T), Val(E)))
    end
    Slink(s.i, s.j, rdr, s.ni, s.linklists)
end

Ilink{T,E,L}(i::Ilink) where {T,E,L} =
    Ilink(padright(i.ndist, zero(T), E), convert(Matrix{Slink{T,E}}, i.slinks))

Lattice{T,E,L,EL}(l::Lattice) where {T,E,L,EL} = 
    Lattice{T,E,L,EL}(l.sublats, l.bravais, l.links)

LinkRules{S}(lr::LinkRules{AutomaticRangeSearch}) where S<:SimpleSearch = LinkRules(SimpleSearch(lr.alg.range), lr.excludesubs, lr.mincells, lr.maxsteps)
LinkRules{T}(lr::LinkRules{AutomaticRangeSearch}) where T<:TreeSearch = LinkRules(TreeSearch(lr.alg.range), lr.excludesubs, lr.mincells, lr.maxsteps)