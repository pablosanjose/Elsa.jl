#######################################################################
# Sublat (sublattice)
#######################################################################
struct Sublat{E,T,D}
    sites::Vector{SVector{E,T}}
    name::NameType
    orbitals::NTuple{D,NameType}
end

Base.empty(s::Sublat) = Sublat(empty(s.sites), s.name, s.orbitals)

Base.copy(s::Sublat) = Sublat(copy(s.sites), s.name, s.orbitals)

Base.show(io::IO, s::Sublat{E,T,D}) where {E,T,D} = print(io,
"Sublat{$E,$T,$D} : sublattice of $T-typed sites in $(E)D space with $D orbitals per site
  Sites    : $(length(s.sites))
  Name     : $(displayname(s))
  Orbitals : $(displayorbitals(s))")

displayname(s::Sublat) = s.name == nametype(:_) ? "pending" : string(":", s.name)
displayorbitals(s::Sublat) = string("(", join(string.(":", s.orbitals), ", "), ")")
nsites(s::Sublat) = length(s.sites)
norbitals(s::Sublat{E,T,D}) where {E,T,D} = D

# API #

"""
    sublat(sites...; name::$(NameType), orbitals = (:noname,))
    sublat(sites::Vector{<:SVector}; name::$(NameType), orbitals = (:noname,))

Create a `Sublat{E,T,D}` that adds a sublattice, of name `name`, with sites at positions
`sites` in `E` dimensional space, each of which hosts `D` different orbitals, with orbital
names specified by `orbitals`. Sites can be entered as tuples or `SVectors`.

# Examples
```jldoctest
julia> sublat((0.0, 0), (1, 1), (1, -1), name = :A, orbitals = (:upspin, :downspin))
Sublat{2,Float64,2} : sublattice in 2D space with 2 orbitals per site
  Sites    : 3
  Name     : :A
  Orbitals : (:upspin, :downspin)
```
"""
sublat(sites::Vector{<:SVector}; name = :_, orbitals = (:noname,)) =
    Sublat(sites, nametype(name), nametype.(Tuple(orbitals)))
sublat(vs::Union{Tuple,AbstractVector{<:Number}}...; kw...) = sublat(toSVectors(vs...); kw...)

transform!(s::S, f::F) where {S <: Sublat,F <: Function} = (s.sites .= f.(s.sites); s)

#######################################################################
# Bravais
#######################################################################
struct Bravais{E,L,T,EL}
    matrix::SMatrix{E,L,T,EL}
end

Bravais{E,T}() where {E,T} = Bravais(SMatrix{E,0,T,0}())

displayvectors(br::Bravais) = displayvectors(br.matrix)
displayvectors(mat::SMatrix{E,L,<:AbstractFloat}; kw...) where {E,L} =
    ntuple(l -> round.(Tuple(mat[:,l]); kw...), Val(L))
displayvectors(mat::SMatrix{E,L,<:Integer}; kw...) where {E,L} =
    ntuple(l -> mat[:,l], Val(L))

Base.show(io::IO, b::Bravais{E,L,T}) where {E,L,T} = print(io,
"Bravais{$E,$L,$T} : set of $L Bravais vectors in $(E)D space.
  Vectors : $(displayvectors(b))
  Matrix  : ", b.matrix)

# API #

"""
    bravais(vecs...)
    bravais(mat)

Create a `Bravais{E,L}` that adds `L` Bravais vectors `vecs` in `E` dimensional space,
alternatively given as the columns of matrix `mat`. For higher instantiation efficiency
enter `vecs` as `Tuple`s or `SVector`s and `mat` as `SMatrix`.

We can scale a `b::Bravais` simply by multiplying it with a factor `a`, like `a * b`.

# Examples
```jldoctest
julia> bravais((1.0, 2), (3, 4))
Bravais{2,2,Float64} : set of 2 Bravais vectors in 2D space.
  Vectors : ((1.0, 2.0), (3.0, 4.0))
  Matrix  : [1.0 3.0; 2.0 4.0]
```
"""
bravais(vs::Union{Tuple, AbstractVector}...) = Bravais(toSMatrix(vs...))

transform(b::Bravais{E,0}, f::F) where {E,F<:Function} = b

function transform(b::Bravais{E,L,T}, f::F) where {E,L,T,F<:Function}
    svecs = let z = zero(SVector{E,T})
        ntuple(i -> f(b.matrix[:, i]) - f(z), Val(L))
    end
    matrix = hcat(svecs...)
    return Bravais(matrix)
end

Base.:*(factor, b::Bravais) = Bravais(factor * b.matrix)
Base.:*(b::Bravais, factor) = Bravais(b.matrix * factor)

#######################################################################
# Unitcell
#######################################################################
struct Unitcell{E,T,O<:Tuple{Vararg{Tuple{Vararg{NameType}}}}}
    sites::Vector{SVector{E,T}}
    names::Vector{NameType}
    offsets::Vector{Int}
    orbitals::O
end

Unitcell(sublats::Sublat...; kw...) = Unitcell(promote(sublats...); kw...)
function Unitcell(sublats::NTuple{N,Sublat{E,T}};
    dim::Val{E2} = Val(E),
    type::Type{T2} = float(T),
    names::Vector{NameType} = [s.name for s in sublats],
    orbitals::NTuple{N,Tuple{Vararg{NameType}}} = (s->s.orbitals).(sublats)) where {N,E,E2,T,T2}
    # Make sublat names unique
    allnames = NameType[:_]
    for i in eachindex(names)
        names[i] in allnames && (names[i] = uniquename(allnames, names[i], i))
        push!(allnames, names[i])
    end
    sites = SVector{E2,T2}[]
    offsets = [0]  # length(offsets) == lenfth(sublats) + 1
    for s in eachindex(sublats)
        append!(sites, sublats[s].sites)
        push!(offsets, length(sites))
    end
    return Unitcell(sites, names, offsets, orbitals)
end

function uniquename(allnames, name, i)
    newname = nametype(Char(64+i)) # Lexicographic, starting from Char(65) = 'A'
    return newname in allnames ? uniquename(allnames, name, i + 1) : newname
end

nsites(u::Unitcell) = length(u.sites)

#######################################################################
# Supercell
#######################################################################
struct Supercell{L,L´,LP,LL´} # LP = L+1, L is lattice dim, L´ is supercell dim
    matrix::SMatrix{L,L´,Int,LL´}
    cellmask::OffsetArray{Bool,LP,BitArray{LP}}
end

Supercell{L,L´}(nsites::Integer,
               ranges::NTuple{L,AbstractRange} = ntuple(_ -> 0:0, Val(L))) where {L,L´} =
    Supercell(SMatrix{L,L´,Int,L*L´}(I),
              OffsetArray(trues(1:nsites, length.(ranges)...), 1:nsites, ranges...))

dim(::Supercell{L,L´}) where {L,L´} = L´

nsites(s::Supercell) = sum(s.cellmask)

function scale(s::Supercell, naxes)
    siterange = first(axes(s.cellmask))
    newmask = similar(s.cellmask, siterange, naxes...)
    bbox = boundingbox(s)
    for c in CartesianIndices(newmask)
        cd, _ = wrap(tail(Tuple(c)), bbox)
        newmask[c] = s.cellmask[first(Tuple(c)), cd...]
    end
    return Supercell(s.matrix, newmask)
end

boundingbox(s::Supercell) = extrema.(tail(axes(s.cellmask)))

@inline function wrap(i::Tuple, bbox)
    n = _wrapdiv.(i, bbox)
    j = _wrapmod.(i, bbox)
    return j, SVector(n)
end

_wrapdiv(n, (nmin, nmax)) = nmin <= n <= nmax ? 0 : div(n - nmin, 1 + nmax - nmin)

_wrapmod(n, (nmin, nmax)) = nmin <= n <= nmax ? n : nmin + mod(n - nmin, 1 + nmax - nmin)

#######################################################################
# Lattice
#######################################################################
struct Lattice{E,L,T<:AbstractFloat,B<:Bravais{E,L,T},U<:Unitcell{E,T},S<:Supercell{L}}
    bravais::B
    unitcell::U
    supercell::S
end
function Lattice(bravais::Bravais{E2,L2}, unitcell::Unitcell{E,T}) where {E2,L2,E,T}
    L = min(E,L2) # L should not exceed E
    Lattice(convert(Bravais{E,L,T}, bravais), unitcell, Supercell{L,L}(nsites(unitcell)))
end

# find SVector type that can hold all orbital amplitudes in any lattice sites
orbitaltype(lat::Lattice{E,L,T}, type::Type{Tv} = Complex{T}) where {E,L,T,Tv} =
    _orbitaltype(SVector{1,Tv}, lat.unitcell.orbitals...)
_orbitaltype(::Type{S}, ::NTuple{D,NameType}, os...) where {N,Tv,D,S<:SVector{N,Tv}} =
    (M = max(N,D); _orbitaltype(SVector{M,Tv}, os...))
_orbitaltype(t) = t

# find SMatrix type that can hold all matrix elements between lattice sites
blocktype(lat::Lattice{E,L,T}, type::Type{Tv} = Complex{T}) where {E,L,T,Tv} =
    _blocktype(orbitaltype(lat, Tv))
_blocktype(::Type{S}) where {N,Tv,S<:SVector{N,Tv}} = SMatrix{N,N,Tv,N*N}

sublat(lat::Lattice, siteidx) = findlast(o -> o < siteidx, lat.unitcell.offsets)
siterange(lat::Lattice, sublat) = (1+lat.unitcell.offsets[sublat]):lat.unitcell.offsets[sublat+1]

displaynames(l::Lattice) = display_as_tuple(l.unitcell.names, ":")
displayorbitals(l::Lattice) = string(l.unitcell.orbitals)
displayvectors(lat::Lattice) = displayvectors(lat.bravais.matrix; digits = 6)

Base.show(io::IO, lat::Lattice{E,L,T}) where {E,L,T} = print(io,
"Lattice{$E,$L,$T} : $(L)D lattice in $(E)D space
  Bravais vectors : $(displayvectors(lat))
  Sublattices     : $(nsublats(lat))
    Names         : $(displaynames(lat))
    Orbitals      : $(displayorbitals(lat))
    Sites         : $(display_as_tuple(sublatsites(lat))) --> $(nsites(lat)) total per unit cell
  Supercell
    Dimensions    : $(dim(lat.supercell))
    Total sites   : $(nsites(lat.supercell))")

# API #

"""
    lattice([bravais::Bravais,] sublats::Sublat...; dim::Val{E}, type::T, names, orbitals)

Create a `Lattice{E,L,T}` with `Bravais` matrix `bravais` and sublattices `sublats`
converted to a common  `E`-dimensional embedding space and type `T`. To override the
embedding  dimension `E`, use keyword `dim = Val(E)`. Similarly, override type `T` with
`type = T`. The keywords `names::Tuple` and `orbitals::Tuple{Tuple}` can be used to rename
`sublats` or redefine their orbitals per site.

# Examples
```jldoctest
julia> lattice(bravais((1, 0)), Sublat((0, 0.)), Sublat((0, Float32(1))); dim = Val(3))
Lattice{3,1,Float64}: 1-dimensional lattice with 2 Float64-typed sublattices in
3-dimensional embedding space
```
"""
lattice(s::Sublat, ss::Sublat...; kw...) where {E,T} = _lattice(Unitcell(s, ss...; kw...))
_lattice(u::Unitcell{E,T}) where {E,T} = Lattice(Bravais{E,T}(), u)
lattice(br::Bravais, s::Sublat, ss::Sublat...; kw...) = Lattice(br, Unitcell(s, ss...; kw...))

# function transform!(lat::Lattice, f::Function; sublats = eachindex(lat.sublats))
#     for s in sublats
#         sind = sublatindex(lat, s)
#         transform!(lat.sublats[sind], f)
#     end
#     br = transform(lat.bravais, f)
#     return Lattice(br, lat.sublats, lat.supercell)
# end

# transform(lat::Lattice, f; kw...) = transform!(deepcopy(lat), f; kw...)
# scale(lat::Lattice, s) =
#     Lattice(lat.bravais, copy.(lat.sublats), scale(lat.supercell, s), copy(lat.offsets))

# Auxiliary #

nsites(lat::Lattice) = length(lat.unitcell.sites)
nsublats(lat) = length(lat.unitcell.names)
sublatsites(lat::Lattice) = diff(lat.unitcell.offsets)

sublatindex(lat::Lattice, name::NameType) = findfirst(s -> (s.name == name), lat.sublats)
sublatindex(lat::Lattice, i::Integer) = Int(i)

#######################################################################
# grow
#######################################################################
const TOOMANYITERS = 10^8

_truefunc(r) = true

"""
    grow(lattice::Lattice{E,L}; supercell = SMatrix{L,0,Int}(), region = r -> true)

Modifies the supercell of an `L`-dimensional lattice to match an `L´`-dimensional
`supercell::SMatrix{L,L´,Int}`. The supercell Bravais matrix `br´` in terms of the `lattice`
Bravais matrix `br` is `br´ = br * supercell`. Only sites at position `r` such that
`region(r) == true` will be included in the supercell. Note that in the case of `L´<L`,
a bounded `region` function must be provided to limit the extension along the non-periodic
dimensions.

`supercell` can be given as an integer matrix `s::SMatrix{L,L2,Int}`, a single integer
`s::Int` (`supercell = s * I`), a single `NTuple{L,Int}` (`supercell` diagonal), or a tuple
of  `NTuple{L,Int}`s (`supercell` columns).

    lattice |> grow(supercell = s, region = f)

Functional syntax, equivalent to `grow(lattice; supercell = s, region = f)

# Examples
```jldoctest
julia> grow(LatticePresets.triangular(), supercell = ((10,2),), region = r-> 0 < r[2] < 4)
Lattice{2,2,Float64} : 2D lattice in 2D space
  Bravais vectors : ((0.5, 0.866025), (-0.5, 0.866025))
  Sublattices     : 1
    Names         : (:1)
    Orbitals      : ((:noname))
    Sites         : (1,) --> 1 total per unit cell
  Supercell
    Dimensions    : 1
    Total sites   : 52
```

# See also

    'Region`
"""
grow(; kw...) = lat -> grow(lat; kw...)
grow(lat::Lattice{E,L}; supercell = SMatrix{L,0,Int}(), region = _truefunc) where {E,L} =
    _grow(lat, tosupercell(supercell, Val(L)), region)

function _grow(lat::Lattice{E,L}, supercell, region) where {E,L}
    bravais = lat.bravais.matrix
    iter = BoxIterator(zero(SVector{L,Int}))
    is_grow_dir = is_perp_dir(supercell)
    foundfirst = false
    counter = 0
    for dn in iter   # We first compute the bounding box
        found = false
        counter += 1; counter == TOOMANYITERS && @warn "`region` seems unbounded"
        is_grow_dn = is_grow_dir(SVector(Tuple(dn)))
        r0 = bravais * SVector(Tuple(dn))
        for s in eachindex(lat.sublats), site in lat.sublats[s].sites
            r = r0 + site
            found = is_grow_dn && region(r)
            if found || !foundfirst
                acceptcell!(iter, dn)
                foundfirst = found
                break
            end
        end
    end
    c = CartesianIndices(iter)
    ns = nsites(lat)
    cellmask = OffsetArray(BitArray(undef, ns, size(c)...), 1:ns, c.indices...)
    @inbounds for dn in c
        is_grow_dn = is_grow_dir(SVector(Tuple(dn)))
        is_grow_dn || (cellmask[:,dn] .= false; continue)
        r0 = bravais * SVector(Tuple(dn))
        counter = 0 # Not threadsafe!
        for s in eachindex(lat.sublats), site in lat.sublats[s].sites
            counter += 1
            r = site + r0
            cellmask[counter, Tuple(dn)...] = is_grow_dn && region(r)
        end
    end
    supercell = Supercell(supercell, cellmask)
    return Lattice(lat.bravais, lat.sublats, supercell)
end

tosupercell(s::SMatrix{L,L2,Int}, ::Val{L}) where {L,L2} = s
tosupercell(s::Number, ::Val{L}) where {L} = SMatrix{L,L,Int}(I)
tosupercell(s::NTuple{L,Number}, ::Val{L}) where {L} =
    SMatrix{L,L,Int}(Diagonal(SVector{L,Int}(s)))
tosupercell(s::NTuple{L2,Tuple}, ::Val{L}) where {L,L2} =
    convert(SMatrix{L,L2,Int}, toSMatrix(s...))

# pseudoinverse of s times an integer n, so that it is an integer matrix (for accuracy)
pinvint(s::SMatrix{N,0}) where {N} = (SMatrix{0,0,Int}(), 0)
function pinvint(s::SMatrix{N,M}) where {N,M}
    qrfact = qr(s)
    pinverse = inv(qrfact.R) * qrfact.Q'
    n = det(qrfact.R)^2
    iszero(n) && throw(ErrorException("Supercell is singular"))
    return round.(Int, n * inv(qrfact.R) * qrfact.Q'), round(Int, n)
end

# This is true whenever old ndist is perpendicular to new lattice
is_perp_dir(supercell) = let invs = pinvint(supercell); dn -> iszero(newndist(dn, invs)); end

newndist(oldndist, (pinvs, n)) = fld.(pinvs * oldndist, n)
newndist(oldndist, (pinvs, n)::Tuple{<:SMatrix{0,0},Int}) = SVector{0,Int}()