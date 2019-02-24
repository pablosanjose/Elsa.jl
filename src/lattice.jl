#######################################################################
# Sublattice (Sublat) : a group of identical sites (e.g. same orbitals)
#######################################################################
"""
    Sublat(sites...; name::$(NameType) = missing, norbitals = 1)
    Sublat(sites::Vector{<:SVector}; name::$(NameType) = missing, norbitals = 1)

Create a `Sublat{T,E}` that adds a sublattice, of
name `name`, with sites at positions `sites` in `E` dimensional space, 
each of which hosts `norbitals` different orbitals. Sites can be entered as tuples
or `SVectors`.

# Examples
```jldoctest
julia> Sublat((0, 0), (1, 1), (1, -1), name = :A)
Sublat{Int64,2}(:A, SArray{Tuple{2},Int64,1,2}[[0, 0], [1, 1], [1, -1]])
```
"""
struct Sublat{T,E}
    sites::Vector{SVector{E,T}}
    norbitals::Int
    name::NameType
end
Sublat(sites::Vector{<:SVector}; name = nametype(missing), norbitals = 1, kw...) = Sublat(sites, norbitals, nametype(name))
Sublat(vs::Union{Tuple,AbstractVector{<:Number}}...; kw...) = Sublat(toSVectors(vs...); kw...)
Sublat{T,E}(;kw...) where {T,E} = Sublat(SVector{E,T}[]; kw...)

dim(s::Sublat) = length(s.sites) * s.norbitals

struct SublatsData
    nsites::Vector{Int}
    norbitals::Vector{Int}
    dims::Vector{Int}
    namesdict::Dict{NameType,Int}
    names::Vector{NameType}
    offsets::Vector{Int}
end
function Base.resize!(sublatsdata::SublatsData, n)
    resize!(sublatsdata.nsites, n)
    resize!(sublatsdata.norbitals, n)
    resize!(sublatsdata.dims, n)
    resize!(sublatsdata.names, n)
    resize!(sublatsdata.offsets, n + 1)
    return sublatsdata
end

SublatsData(sublats) = sublatsdata!(SublatsData(), sublats)
SublatsData() = SublatsData(Int[], Int[], Int[], Dict{NameType,Int}(), NameType[], Int[])
function sublatsdata!(sublatsdata, sublats)
    resize!(sublatsdata, length(sublats))
    sublatsdata.offsets[1] = 0
    for (i, sublat) in enumerate(sublats)
        actualname = sublat.name == nametype(missing) ? nametype("S$i") : sublat.name
        haskey(sublatsdata.namesdict, actualname) && (actualname = nametype("$(actualname)$i"))
        haskey(sublatsdata.namesdict, actualname) && throw(ErrorException("Duplicate sublattice name"))
        sublatsdata.namesdict[actualname] = i
        sublatsdata.names[i] = actualname
        sublatsdata.norbitals[i] = sublat.norbitals
        sublatsdata.nsites[i] = length(sublat.sites)
        sublatsdata.dims[i] = dim(sublat)
        sublatsdata.offsets[i+1] = sublatsdata.offsets[i] + sublatsdata.dims[i]
    end
    return sublatsdata
end

nsublats(s::SublatsData) = length(s.dims)

sublatindex(s::SublatsData, name::NameType) = s.namesdict[name]
sublatindex(s::SublatsData, i::Integer) = Int(i)
Base.keys(s::SublatsData) = keys(s.nsites)

function tosite(row, sublatsdata)
    s = findsublat(row, sublatsdata.offsets)
    offset = sublatsdata.offsets[s]
    norbs = sublatsdata.norbitals[s]
    delta = row - offset
    return div(delta, norbs), rem(delta, norbs), s
end

torow(siteindex, s, sublatsdata) = sublatsdata.offsets[s] + (siteindex - 1) * sublatsdata.norbitals[s] + 1

function findsublat(row, offsets)
    for n in eachindex(offsets)
        offsets[n] >= row && return n - 1
    end
    return 0
end

transform!(s::S, f::F) where {S <: Sublat,F <: Function} = (s.sites .= f.(s.sites); s)

# flatten(ss::Sublat...) = Sublat(ss[1].name, vcat((s.sites for s in ss)...))

# function sublatoffsets(sublats::Vector{<:Sublat})
#     ns = length(sublats)
#     offsets = Vector{Int}(undef, ns + 1)
#     offset = 1
#     for no in 1:ns
#         offsets[no] = offset
#         offset += sublats[no].norbitals * length(sublats[no].sites)
#     end
#     offsets[ns + 1] = offsets
#     return offsets
# end

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