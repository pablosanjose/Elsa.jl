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
    orbitals::NTuple{N,Tuple} = (s->s.orbitals).(sublats)) where {N,E,E2,T,T2}
    # Make sublat names unique
    allnames = NameType[:_]
    for i in eachindex(names)
        names[i] in allnames && (names[i] = uniquename(allnames, names[i], i))
        push!(allnames, names[i])
    end
    sites = SVector{E2,T2}[]
    offsets = [0]  # length(offsets) == length(sublats) + 1
    for s in eachindex(sublats)
        for site in sublats[s].sites
            push!(sites, padright(site, Val(E2)))
        end
        push!(offsets, length(sites))
    end
    return Unitcell(sites, names, offsets, (n -> nametype.(n)).(orbitals))
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

Supercell{L}(ns::Integer) where {L} =
    Supercell(SMatrix{L,L,Int,L*L}(I), trues(1:ns, ntuple(_->0:0, Val(L))...))

dim(::Supercell{L,L´}) where {L,L´} = L´

nsites(s::Supercell) = sum(s.cellmask)

Base.CartesianIndices(s::Supercell) = CartesianIndices(tail(axes(s.cellmask)))

function Base.show(io::IO, s::Supercell{L,L´}) where {L,L´}
    i = get(io, :indent, "")
    print(io,
"$(i)Supercell{$L,$(L´)} for $(L´)D superlattice of the base $(L)D lattice
$(i)  Supervectors  : $(displayvectors(s.matrix))
$(i)  Supersites    : $(nsites(s))")
end

#######################################################################
# Lattice
#######################################################################
struct Lattice{E,L,T<:AbstractFloat,S<:Supercell{L},B<:Bravais{E,L,T},U<:Unitcell{E,T}}
    bravais::B
    unitcell::U
    supercell::S
end
function Lattice(bravais::Bravais{E2,L2}, unitcell::Unitcell{E,T}) where {E2,L2,E,T}
    L = min(E,L2) # L should not exceed E
    Lattice(convert(Bravais{E,L,T}, bravais), unitcell, Supercell{L}(nsites(unitcell)))
end

displaynames(l::Lattice) = display_as_tuple(l.unitcell.names, ":")
displayorbitals(l::Lattice) =
    replace(replace(string(l.unitcell.orbitals), "Symbol(\"" => ":"), "\")" => "")

function Base.show(io::IO, lat::Lattice{E,L,T}) where {E,L,T}
    i = get(io, :indent, "")
    ioindent = IOContext(io, :indent => "$(i)  ")
    print(io,
"$(i)Lattice{$E,$L,$T} : $(L)D lattice in $(E)D space
$(i)  Bravais vectors : $(displayvectors(lat.bravais.matrix; digits = 6))
$(i)  Sublattices     : $(nsublats(lat))
$(i)    Names         : $(displaynames(lat))
$(i)    Orbitals      : $(displayorbitals(lat))
$(i)    Sites         : $(display_as_tuple(sublatsites(lat))) --> $(nsites(lat)) total per unit cell")
    isminimalsupercell(lat) || print(ioindent, "\n", lat.supercell)
end

# API #

"""
    lattice([bravais::Bravais,] sublats::Sublat...; dim::Val{E}, type::T, names, orbitals)

Create a `Lattice{E,L,T}` with Bravais matrix `bravais` and sublattices `sublats`
converted to a common  `E`-dimensional embedding space and type `T`. To override the
embedding  dimension `E`, use keyword `dim = Val(E)`. Similarly, override type `T` with
`type = T`. The keywords `names::Tuple` and `orbitals::Tuple{Tuple}` can be used to rename
`sublats` or redefine their orbitals per site.

    lattice(superlat)

Create a lattice with a unitcell equal to the supercell of superlattice `superlat`.
See also `superlattice` for information on building superlattices.

# Examples
```jldoctest
julia> lattice(bravais((1, 0)), sublat((0, 0)), sublat((0, Float32(1))); dim = Val(3))
Lattice{3,1,Float32} : 1D lattice in 3D space
  Bravais vectors : ((1.0, 0.0, 0.0),)
  Sublattices     : 2
    Names         : (:A, :B)
    Orbitals      : ((:noname,), (:noname,))
    Sites         : (1, 1) --> 2 total per unit cell

julia> lattice(superlattice(LatticePresets.honeycomb(), 100))
Lattice{2,2,Float64} : 2D lattice in 2D space
  Bravais vectors : ((0.5, 0.866025), (-0.5, 0.866025))
  Sublattices     : 2
    Names         : (:A, :B)
    Orbitals      : ((:noname,), (:noname,))
    Sites         : (10000, 10000) --> 20000 total per unit cell
```
"""
lattice(s::Sublat, ss::Sublat...; kw...) where {E,T} = _lattice(Unitcell(s, ss...; kw...))
_lattice(u::Unitcell{E,T}) where {E,T} = Lattice(Bravais{E,T}(), u)
lattice(br::Bravais, s::Sublat, ss::Sublat...; kw...) = Lattice(br, Unitcell(s, ss...; kw...))

function lattice(lat::Lattice)
    newoffsets = supercell_offsets(lat)
    newsites = supercell_sites(lat)
    unitcell = Unitcell(newsites, lat.unitcell.names, newoffsets, lat.unitcell.orbitals)
    bravais = lat.bravais * lat.supercell.matrix
    return Lattice(bravais, unitcell)
end

function supercell_offsets(lat::Lattice)
    sitecounts = zeros(Int, nsublats(lat) + 1)
    foreach_supersite((s, oldi, dn, newi) -> sitecounts[s + 1] += 1, lat)
    newoffsets = cumsum!(sitecounts, sitecounts)
    return newoffsets
end

function supercell_sites(lat::Lattice)
    newsites = similar(lat.unitcell.sites, nsites(lat.supercell))
    oldsites = lat.unitcell.sites
    bravais = lat.bravais.matrix
    foreach_supersite((s, oldi, dn, newi) -> newsites[newi] = bravais * dn + oldsites[oldi], lat)
    return newsites
end

# Auxiliary #

# find SVector type that can hold all orbital amplitudes in any lattice sites
orbitaltype(lat::Lattice{E,L,T}, type::Type{Tv} = Complex{T}) where {E,L,T,Tv} =
    _orbitaltype(SVector{1,Tv}, lat.unitcell.orbitals...)
_orbitaltype(::Type{S}, ::NTuple{D,NameType}, os...) where {N,Tv,D,S<:SVector{N,Tv}} =
    (M = max(N,D); _orbitaltype(SVector{M,Tv}, os...))
_orbitaltype(t::Type{SVector{N,Tv}}) where {N,Tv} = t
_orbitaltype(t::Type{SVector{1,Tv}}) where {Tv} = Tv

# find SMatrix type that can hold all matrix elements between lattice sites
blocktype(lat::Lattice{E,L,T}, type::Type{Tv} = Complex{T}) where {E,L,T,Tv} =
    _blocktype(orbitaltype(lat, Tv))
_blocktype(::Type{S}) where {N,Tv,S<:SVector{N,Tv}} = SMatrix{N,N,Tv,N*N}
_blocktype(::Type{S}) where {S<:Number} = S

sitetype(::Lattice{E,L,T}) where {E,L,T} = T

sublat(lat::Lattice, siteidx) = findlast(o -> o < siteidx, lat.unitcell.offsets)
siterange(lat::Lattice, sublat) = (1+lat.unitcell.offsets[sublat]):lat.unitcell.offsets[sublat+1]

sublatsites(lat::Lattice) = diff(lat.unitcell.offsets)

nsites(lat::Lattice) = length(lat.unitcell.sites)
nsites(lat::Lattice, sublat) = sublatsites(lat)[sublat]

nsublats(lat) = length(lat.unitcell.names)

# apply f to trues in cellmask. Arguments are oldi = old site index, s = sublat index
function foreach_supersite(f::F, lat::Lattice) where {F<:Function}
    newi = 0
    for s in 1:nsublats(lat), oldi in siterange(lat, s)
        for dn in CartesianIndices(lat.supercell)
            if lat.supercell.cellmask[oldi, Tuple(dn)...]
                newi += 1
                f(s, oldi, SVector(Tuple(dn)), newi)
            end
        end
    end
    return nothing
end

isminimalsupercell(lat::Lattice{E,L,T,S}) where {E,L,T,S<:Supercell{L,L}} =
    all(r -> r == 0:0, tail(cellmaskaxes(lat))) && all(lat.supercell.cellmask)
isminimalsupercell(lat::Lattice{E,L,T,S}) where {E,L,T,S} = false

# sublatindex(lat::Lattice, name::NameType) = findfirst(s -> (s.name == name), lat.sublats)
# sublatindex(lat::Lattice, i::Integer) = Int(i)

#######################################################################
# superlattice
#######################################################################
const TOOMANYITERS = 10^8

"""
    superlattice(lattice::Lattice{E,L}, v::NTuple{L,Integer}...; region = missing)
    superlattice(lattice::Lattice{E,L}, supercell::SMatrix{L,L´,Int}; region = missing)

Modifies the supercell of `L`-dimensional `lattice` to one with Bravais vectors
`br´= br * supercell`, where `supercell::SMatrix{L,L´,Int}` is the integer supercell matrix
with the `L´` `v`s as columns.

Only sites at position `r` such that `region(r) == true` will be included in the supercell.
If `region` is missing, a Bravais unit cell perpendicular to the `v` axes will be selected
for the `L-L´` non-periodic directions.

    superlattice(lattice::Lattice{E,L}, factor::Integer; region = missing)

Calls `superlattice` with a uniformly scaled `supercell = SMatrix{L,L}(factor * I)`

    superlattice(lattice::Lattice{E,L}, factors::Integer...; region = missing)

Calls `superlattice` with different scaling along each Bravais vector (diagonal supercell
with factors along the diagonal)

    lattice |> superlattice(v...; kw...)

Functional syntax, equivalent to `superlattice(lattice, v...; kw...)

# Examples
```jldoctest
julia> superlattice(LatticePresets.honeycomb(), region = RegionPresets.circle(300))
Lattice{2,2,Float64} : 2D lattice in 2D space
  Bravais vectors : ((0.5, 0.866025), (-0.5, 0.866025))
  Sublattices     : 2
    Names         : (:A, :B)
    Orbitals      : ((:noname,), (:noname,))
    Sites         : (1, 1) --> 2 total per unit cell
  Supercell
    Dimensions    : 0
    Vectors       : ()
    Total sites   : 652966

julia> superlattice(LatticePresets.triangular(), (1,1), (1, -1))
Lattice{2,2,Float64} : 2D lattice in 2D space
  Bravais vectors : ((0.5, 0.866025), (-0.5, 0.866025))
  Sublattices     : 1
    Names         : (:A)
    Orbitals      : ((:noname,),)
    Sites         : (1) --> 1 total per unit cell
  Supercell
    Dimensions    : 2
    Vectors       : ((1, 1), (1, -1))
    Total sites   : 2

julia> LatticePresets.square() |> superlattice(3)
Lattice{2,2,Float64} : 2D lattice in 2D space
  Bravais vectors : ((1.0, 0.0), (0.0, 1.0))
  Sublattices     : 1
    Names         : (:A)
    Orbitals      : ((:noname,),)
    Sites         : (1) --> 1 total per unit cell
  Supercell
    Dimensions    : 2
    Vectors       : ((3, 0), (0, 3))
    Total sites   : 9
```
"""
superlattice(v::Union{SMatrix,Tuple,SVector}...; kw...) = lat -> superlattice(lat, v...; kw...)
superlattice(lat::Lattice{E,L}; kw...) where {E,L} = superlattice(lat, SMatrix{L,0,Int}(); kw...)
superlattice(lat::Lattice{E,L}, factor::Integer; kw...) where {E,L} =
    superlattice(lat, SMatrix{L,L,Int}(factor * I); kw...)
superlattice(lat::Lattice{E,L}, factors::Vararg{Integer,L}; kw...) where {E,L} =
    superlattice(lat, SMatrix{L,L,Int}(Diagonal(SVector(factors))); kw...)
superlattice(lat::Lattice{E,L}, vecs::NTuple{L,Int}...; kw...) where {E,L} =
    superlattice(lat, toSMatrix(Int, vecs...); kw...)
function superlattice(lat::Lattice{E,L}, supercell::SMatrix{L,L´,Int}; region = missing) where {E,L,L´}
    bravais = lat.bravais.matrix
    regionfunc = region === missing ? ribbonfunc(bravais, supercell) : region
    iter = BoxIterator(zero(SVector{L,Int}))
    in_supercell_func = is_perp_dir(supercell)
    foundfirst = false
    counter = 0
    for dn in iter   # We first compute the bounding box
        found = false
        counter += 1; counter == TOOMANYITERS &&
            throw(ArgumentError("`region` seems unbounded (after $TOOMANYITERS iterations)"))
        in_supercell = in_supercell_func(SVector(Tuple(dn)))
        r0 = bravais * SVector(Tuple(dn))
        for site in lat.unitcell.sites
            r = r0 + site
            found = in_supercell && regionfunc(r)
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
        dntup = Tuple(dn)
        dnvec = SVector(dntup)
        in_supercell = in_supercell_func(dnvec)
        in_supercell || (cellmask[:, dntup...] .= false; continue)
        r0 = bravais * dnvec
        for (i, site) in enumerate(lat.unitcell.sites)
            r = site + r0
            cellmask[i, dntup...] = in_supercell && regionfunc(r)
        end
    end
    supercell = Supercell(supercell, cellmask)
    return Lattice(lat.bravais, lat.unitcell, supercell)
end

# pinvmultiple(::Missing) = (I, 1)
# pinvmultiple(s::Supercell) = pinvmultiple(s.matrix)

# This is true whenever old ndist is perpendicular to new lattice
is_perp_dir(supercell) = let invs = pinvmultiple(supercell); dn -> iszero(new_dn(dn, invs)); end

new_dn(oldndist, (pinvs, n)) = fld.(pinvs * oldndist, n)
new_dn(oldndist, ::Tuple{<:SMatrix{0,0},Int}) = SVector{0,Int}()

wrap_dn(olddn::SVector, newdn::SVector, supercell::SMatrix) = olddn - supercell * newdn
# wrap_dn(olddn::SVector, newdn::SVector, ::Missing) = zero(olddn)

# @inline function wrap(i::Tuple, bbox)
#     n = _wrapdiv.(i, bbox)
#     j = _wrapmod.(i, bbox)
#     return j, SVector(n)
# end

# _wrapdiv(n, (nmin, nmax)) = nmin <= n <= nmax ? 0 : div(n - nmin, 1 + nmax - nmin)

# _wrapmod(n, (nmin, nmax)) = nmin <= n <= nmax ? n : nmin + mod(n - nmin, 1 + nmax - nmin)


function ribbonfunc(bravais::SMatrix{E,L,T}, supercell::SMatrix{L,L´}) where {E,L,T,L´}
    L <= L´ && return truefunc
    # real-space supercell axes + all space
    s = hcat(bravais * supercell, SMatrix{E,E,T}(I))
    q = qr(s).Q
    # last vecs in Q are orthogonal to axes
    os = ntuple(i -> SVector(q[:,i+L´]), Val(E-L´))
    # range of projected bravais, including zero
    ranges = (o -> extrema(vcat(SVector(zero(T)), bravais' * o)) .- sqrt(eps(T))).(os)
    # projector * r gives the projection of r on orthogonal vectors
    projector = hcat(os...)'
    # ribbon defined by r's with projection within ranges for all orthogonal vectors
    regionfunc(r) = all(first.(ranges) .<= Tuple(projector * r) .< last.(ranges))
    # return r -> all(first.(ranges) .<= Tuple(projector * r) .<= last.(ranges))
    return regionfunc
end
truefunc(r) = true