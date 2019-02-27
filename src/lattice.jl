#######################################################################
# Sublattice (Sublat) : a group of identical sites (e.g. same orbitals)
#######################################################################
"""
    Sublat(sites...; name::$(NameType))
    Sublat(sites::Vector{<:SVector}; name::$(NameType))

Create a `Sublat{E,T}` that adds a sublattice, of
name `name`, with sites at positions `sites` in `E` dimensional space, 
each of which hosts `norbitals` different orbitals. Sites can be entered as tuples
or `SVectors`.

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
Sublat(vs::Union{Tuple,AbstractVector{<:Number}}...; kw...) = 
    Sublat(toSVectors(vs...); kw...)
Sublat{E,T}(;kw...) where {E,T} = Sublat(SVector{E,T}[]; kw...)

Base.show(io::IO, s::Sublat{E,T}) where {E,T} = print(io, 
"Sublat{$T,$E}: sublattice `:$(s.name)` of $(length(s.sites)) $T-typed sites in $E-dimensional embedding space")

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
struct Bravais{E,L,EL}
    matrix::SMatrix{E,L,Float64,EL}
end

Bravais{E}() where {E} = Bravais(SMatrix{E,0,Float64,0}())
Bravais(vs::Union{Tuple, AbstractVector}...) = Bravais(toSMatrix(Float64, vs...))
Bravais(s::SMatrix{N,M}) where {N,M} = Bravais(convert(SMatrix{N,M,Float64}, s))

transform(b::Bravais{E,0,0}, f::F) where {E,F <: Function} = b
function transform(b::Bravais{E,L,EL}, f::F) where {E,L,EL,F<:Function}
    svecs = let z = zero(SVector{E,Float64})
        ntuple(i -> f(b.matrix[:, i]) - f(z), Val(L))
    end
    matrix = hcat(svecs...)
    return Bravais(matrix)
end

#######################################################################
# Lattice
#######################################################################
mutable struct Lattice{E,L,T,EL}  # Lattice transform needs to change bravais
    sublats::Vector{Sublat{E,T}}
    bravais::Bravais{E,L,EL}
end
Lattice(sublats::Sublat{E}...; kw...) where {E} = Lattice(Bravais{E}(), sublats...; kw...)
Lattice(bravais::Bravais, sublats::Sublat...; kw...) = 
    _lattice(bravais, promote(sublats...); kw...)
function _lattice(
        bravais::Bravais{E,L}, 
        sublats::Union{NTuple{N,Sublat{E,T}},Vector{Sublat{E,T}}}; 
        dim::Val{E2} = Val(E), ptype::Type{T2} = T, kw...) where {N,T,E,L,T2,E2}
    actualsublats = convert(Vector{Sublat{E2,T2}}, collect(sublats))
    actualbravais = convert(Bravais{E2,L}, bravais)
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