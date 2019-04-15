# In v0.7+ the idea is that `convert`` is a shortcut to a "safe" subset of the constructors 
# for a type, that guarantees the resulting type. The constructor is the central machinery
# to instantiate types and convert between instances. (For parametric types, unless overridden, 
# you get an implicit internal constructor without parameters, so no need to define that externally)

convert(::Type{T}, l::T) where T<:Bravais = l
convert(::Type{T}, l::Bravais) where T<:Bravais = T(l)

convert(::Type{T}, l::T) where T<:Sublat = l
convert(::Type{T}, l::Sublat) where T<:Sublat = T(l)

convert(::Type{T}, l::T) where T<:System = l
convert(::Type{T}, l::System) where T<:System = T(l)

convert(::Type{T}, l::T) where T<:Operator = l
convert(::Type{T}, l::Operator) where T<:Operator = T(l)

convert(::Type{T}, l::T) where T<:Block = l
convert(::Type{T}, l::Block) where T<:Block = T(l)

convert(::Type{T}, l::T) where T<:Model = l
convert(::Type{T}, l::Model) where T<:Model = T(l)

# Constructors for conversion

Sublat{E,T}(s::Sublat) where {E,T} = 
    Sublat([padright(site, zero(T), Val(E)) for site in s.sites], s.name)

Base.promote_rule(::Type{Sublat{E1,T1}}, ::Type{Sublat{E2,T2}}) where {E1,E2,T1,T2} = 
    Sublat{max(E1, E2), promote_type(T1, T2)}

Bravais{E,L,T}(b::Bravais) where {E,L,T} = 
    Bravais(padrightbottom(b.matrix, SMatrix{E,L,T}))

System{E,L,T,Tv}(s::System) where {E,L,T,Tv} = 
    System(convert(Lattice{E,L,T,Tv}, s.lattice), Operator{Tv,L}(s.hamiltonian))

Operator{Tv,L}(o::Operator) where {Tv,L} = 
    Operator{Tv,L}(o.matrix, o.intra, o.inters, o.boundary)

Block{Tv,L}(b::Block) where {Tv,L} = 
    Block{Tv,L}(b.ndist, b.matrix, b.sysinfo, b.nlinks)

Model{Tv}(m::Model) where {Tv} = Model{Tv}(m.terms...)
promote_model(model::Model, sys::System{E,L,T,Tv}, systems...) where {E,L,T,Tv} = promote_model(Tv, model, systems...)
promote_model(::Type{Tv}, model::Model, sys::System{E,L,T,Tv2}, systems...) where {Tv,E,L,T,Tv2} = promote_model(promote_type(Tv, Tv2), model, systems...)
promote_model(::Type{Tv}, model::Model) where {Tv} = convert(Model{Tv}, model)