#######################################################################
# Sublattice (Sublat) : a group of identical sites (e.g. same orbitals)
#######################################################################
"""
    Sublat(sites...; name::$(NameType))
    Sublat(sites::Vector{<:SVector}; name::$(NameType))

Create a `Sublat{E,T}` that adds a sublattice, of name `name`, with sites at positions 
`sites` in `E` dimensional space, each of which hosts `norbitals` different orbitals. Sites 
can be entered as tuples or `SVectors`.

# Examples
```jldoctest
julia> Sublat((0, 0), (1, 1), (1, -1), name = :A)
Sublat{Int64,2}(:A, SArray{Tuple{2},Int64,1,2}[[0, 0], [1, 1], [1, -1]])
```
"""
mutable struct Sublat{E,T}  # Mutable because name is non-isbits and slow anyway
    sites::Vector{SVector{E,T}}
    name::NameType
end

Sublat(sites::Vector{<:SVector}; name = :_, kw...) = Sublat(sites, name)
Sublat(vs::Union{Tuple,AbstractVector{<:Number}}...; kw...) = Sublat(toSVectors(vs...); kw...)
Sublat{E,T}(;kw...) where {E,T} = Sublat(SVector{E,T}[]; kw...)

Base.show(io::IO, s::Sublat{E,T}) where {E,T} = print(io, 
"Sublat{$E,$T}: sublattice `:$(s.name)` of $(length(s.sites)) $T-typed sites in $E-dimensional embedding space")

transform!(s::S, f::F) where {S <: Sublat,F <: Function} = (s.sites .= f.(s.sites); s)

#######################################################################
# Bravais
#######################################################################
"""
    Bravais(vecs...)
    Bravais(mat)

Create a `Bravais{E,L}` that adds `L` Bravais vectors
`vecs` in `E` dimensional space, alternatively given as the columns of matrix
`mat`. For higher efficiency write `vecs` as `Tuple`s or `SVector`s and `mat`
as `SMatrix`.

# Examples
```jldoctest
julia> Bravais((1, 2), (3, 4))
Bravais{Int64,2,2,4}([1 3; 2 4])
```
"""
struct Bravais{E,L,T,EL}
    matrix::SMatrix{E,L,T,EL}
end

Bravais{E}() where {E} = Bravais(SMatrix{E,0,Float64,0}())
Bravais(vs::Union{Tuple, AbstractVector}...) = Bravais(toSMatrix(vs...))

transform(b::Bravais{E,0}, f::F) where {E,F <: Function} = b

function transform(b::Bravais{E,L}, f::F) where {E,L,F<:Function}
    svecs = let z = zero(SVector{E,Float64})
        ntuple(i -> f(b.matrix[:, i]) - f(z), Val(L))
    end
    matrix = hcat(svecs...)
    return Bravais(matrix)
end

#######################################################################
# Lattice
#######################################################################
"""
    Lattice(bravais::Bravais, sublats::Sublat...; dim::Val{E}, ptype::T)

Create a `Lattice{E,L,T}` with `Bravais` matrix `bravais` and sublattices `sublats` in 
`E`-dimensional space, converted to a common type `Sublat{E,T}`. `bravais` is converted 
to match `E` and `T` from `sublats`. To override the embedding  dimension `E`, use keyword 
`dim = Val(E)`. Similarly, override type `T` with `ptype = T`.

# Examples
```jldoctest
julia> Lattice(Bravais((1, 0)), Sublat((0, 0.)), Sublat((0, Float32(1))); dim = Val(3))
Lattice{3,1,Float64}: 1-dimensional lattice with 2 Float64-typed sublattices in 
3-dimensional embedding space
```
"""
mutable struct Lattice{E,L,T,EL}  # mutable: Lattice transform needs to change bravais
    sublats::Vector{Sublat{E,T}}
    bravais::Bravais{E,L,T,EL}
end

Lattice(sublats::Sublat{E}...; kw...) where {E} = Lattice(Bravais{E}(), sublats...; kw...)
Lattice(bravais::Bravais, sublats::Sublat...; kw...) = 
    _lattice(bravais, promote(sublats...); kw...)

function _lattice( 
        bravais::Bravais{EB,L},
        sublats::Union{NTuple{N,Sublat{E,T}},Vector{Sublat{E,T}}}; 
        dim::Val{E2} = Val(E), ptype::Type{T2} = T, kw...) where {N,T,E,L,T2,E2,EB}
    actualsublats = convert(Vector{Sublat{E2,T2}}, collect(sublats))
    actualbravais = convert(Bravais{E2,L,T2}, bravais)
    names = NameType[:_]
    for (i, sublat) in enumerate(actualsublats)
        if sublat.name in names
            actualname = uniquename(names, sublat.name, i)
            actualsublats[i].name = actualname
        else
            actualname = sublat.name
        end
        push!(names, actualname)
    end
    return Lattice(actualsublats, actualbravais)
end

function uniquename(names, name, i)
    newname = Symbol(:_, i)
    return newname in names ? uniquename(names, name, i + 1) : newname
end

Base.show(io::IO, s::Lattice{E,L,T}) where {E,L,T} = print(io, 
"Lattice{$E,$L,$T}: $L-dimensional lattice with $(length(s.sublats)) $T-typed $(length(s.sublats) > 1 ? 
"sublattices" : "sublattice") in $E-dimensional embedding space")
