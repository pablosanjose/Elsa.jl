# In v0.7+ the idea is that `convert`` is a shortcut to a "safe" subset of the constructors 
# for a type, that guarantees the resulting type. The constructor is the central machinery
# to instantiate types and convert between instances. (For parametric types, unless overridden, 
# you get an implicit internal constructor without parameters, so no need to define that externally)

convert(::Type{T}, l::T) where T<:Lattice = l
convert(::Type{T}, l::Lattice) where T<:Lattice = T(l)

convert(::Type{T}, l::T) where T<:LatticeDirective = l
convert(::Type{T}, l::LatticeDirective) where T<:LatticeDirective = T(l)

convert(::Type{T}, l::T) where T<:Links = l
convert(::Type{T}, l::Links) where T<:Links = T(l)
convert(::Type{T}, l::T) where T<:Ilink = l
convert(::Type{T}, l::Ilink) where T<:Ilink = T(l)
convert(::Type{T}, l::T) where T<:Slink = l
convert(::Type{T}, l::Slink) where T<:Slink = T(l)

# Constructors for conversion

Sublat{T,E}(s::Sublat) where {T,E} = 
    Sublat(s.name, [padright(site, zero(T), Val(E)) for site in s.sites])

Bravais{T,E,L,EL}(b::Bravais) where {T,E,L,EL} = 
    Bravais(padrightbottom(b.matrix, SMatrix{E,L,T,EL}))

function Slink{T,E}(s::Slink) where {T,E}
    nzval = Tuple{SVector{E,T}, SVector{E,T}}[(padright(r, zero(T), Val(E)), padright(dr, zero(T), Val(E))) for (r, dr) in s.rdr.nzval]
    Slink(SparseMatrixCSC(s.rdr.m, s.rdr.n, s.rdr.colptr, s.rdr.rowval, nzval))
end

Links{T,E,L}(l::Links) where {T,E,L} = 
    Links{T,E,L}(l.intralink, l.interlinks)

Ilink{T,E,L}(i::Ilink) where {T,E,L} =
    Ilink(padright(i.ndist, zero(Int), Val(L)), convert(Matrix{Slink{T,E}}, i.slinks))

Lattice{T,E,L,EL}(l::Lattice) where {T,E,L,EL} = 
    Lattice{T,E,L,EL}(l.sublats, l.bravais, l.links)

LinkRule{S}(lr::LinkRule{AutomaticRangeSearch}) where S<:SimpleSearch = LinkRule(SimpleSearch(lr.alg.range), lr.sublats, lr.mincells, lr.maxsteps)
LinkRule{T}(lr::LinkRule{AutomaticRangeSearch}) where T<:TreeSearch = LinkRule(TreeSearch(lr.alg.range), lr.sublats, lr.mincells, lr.maxsteps)