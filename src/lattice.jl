abstract type AbstractLattice{E,L,T} end

#######################################################################
# Sublat (sublattice)
#######################################################################
struct Sublat{E,T}
    sites::Vector{SVector{E,T}}
    name::NameType
end

Base.empty(s::Sublat) = Sublat(empty(s.sites), s.name)

Base.copy(s::Sublat) = Sublat(copy(s.sites), s.name)

Base.show(io::IO, s::Sublat{E,T}) where {E,T} = print(io,
"Sublat{$E,$T,$D} : sublattice of $T-typed sites in $(E)D space
  Sites    : $(length(s.sites))
  Name     : $(displayname(s))")

displayname(s::Sublat) = s.name == nametype(:_) ? "pending" : string(":", s.name)
# displayorbitals(s::Sublat) = string("(", join(string.(":", s.orbitals), ", "), ")")
nsites(s::Sublat) = length(s.sites)

# External API #

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
sublat(sites::Vector{<:SVector}; name = :_) =
    Sublat(sites, nametype(name))
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

# External API #

"""
    bravais(vecs...)
    bravais(matrix)

Create a `Bravais{E,L}` that adds `L` Bravais vectors `vecs` in `E` dimensional space,
alternatively given as the columns of matrix `mat`. For higher instantiation efficiency
enter `vecs` as `Tuple`s or `SVector`s and `mat` as `SMatrix`.

We can scale a `b::Bravais` simply by multiplying it with a factor `a`, like `a * b`.

    bravais(lat::Lattice)
    bravais(h::Hamiltonian)

Obtain the Bravais matrix of lattice `lat` or Hamiltonian `h`

# Examples
```jldoctest
julia> bravais((1.0, 2), (3, 4))
Bravais{2,2,Float64} : set of 2 Bravais vectors in 2D space.
  Vectors : ((1.0, 2.0), (3.0, 4.0))
  Matrix  : [1.0 3.0; 2.0 4.0]
```
"""
bravais(vs::Union{Tuple, AbstractVector}...) = Bravais(toSMatrix(vs...))
bravais(lat::AbstractLattice) = lat.bravais.matrix

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
struct Unitcell{E,T,N}
    sites::Vector{SVector{E,T}}
    names::NTuple{N,NameType}
    offsets::Vector{Int}
end

Unitcell(sublats::Sublat...; kw...) = Unitcell(promote(sublats...); kw...)
function Unitcell(sublats::NTuple{N,Sublat{E,T}};
    dim::Val{E2} = Val(E),
    type::Type{T2} = float(T),
    names = [s.name for s in sublats]) where {N,E,E2,T,T2}
    _names = nametype_vector(names, N)
    # Make sublat names unique
    allnames = NameType[:_]
    for i in eachindex(_names)
        _names[i] in allnames && (_names[i] = uniquename(allnames, _names[i], i))
        push!(allnames, _names[i])
    end
    sites = SVector{E2,T2}[]
    offsets = [0]  # length(offsets) == length(sublats) + 1
    for s in eachindex(sublats)
        for site in sublats[s].sites
            push!(sites, padright(site, Val(E2)))
        end
        push!(offsets, length(sites))
    end
    return Unitcell(sites, sanitize_names(_names, Val(N)), offsets)
end

nametype_vector(names::AbstractVector, ::Integer) = nametype.(names)
nametype_vector(names::Tuple, ::Integer) = [nametype.(names)...]
nametype_vector(name::Union{NameType,Int}, N::Integer) = fill(name, N)
sanitize_names(names::Vector, ::Val{N}) where {N} = ntuple(n -> names[n], Val(N))
# sanitize_orbs(o::NTuple{N,Tuple}, ::Val{N}) where {N} = (n -> nametype.(n)).(o)
# sanitize_orbs(o::NTuple{M,<:Any}, ::Val{N}) where {M,N} =
#     ntuple(n -> ntuple(m -> nametype(o[m]), Val(M)), Val(N))

function uniquename(allnames, name, i)
    newname = nametype(Char(64+i)) # Lexicographic, starting from Char(65) = 'A'
    return newname in allnames ? uniquename(allnames, name, i + 1) : newname
end

nsites(u::Unitcell) = length(u.sites)

#######################################################################
# Lattice
#######################################################################
struct Lattice{E,L,T<:AbstractFloat,B<:Bravais{E,L,T},U<:Unitcell{E,T}} <: AbstractLattice{E,L,T}
    bravais::B
    unitcell::U
end
function Lattice(bravais::Bravais{E2,L2}, unitcell::Unitcell{E,T}) where {E2,L2,E,T}
    L = min(E,L2) # L should not exceed E
    Lattice(convert(Bravais{E,L,T}, bravais), unitcell)
end

displaynames(l::AbstractLattice) = display_as_tuple(l.unitcell.names, ":")
# displayorbitals(l::AbstractLattice) =
#     replace(replace(string(l.unitcell.orbitals), "Symbol(\"" => ":"), "\")" => "")

function Base.show(io::IO, lat::Lattice)
    i = get(io, :indent, "")
    print(io, i, summary(lat), "\n",
"$i  Bravais vectors : $(displayvectors(lat.bravais.matrix; digits = 6))
$i  Sublattices     : $(nsublats(lat))
$i    Names         : $(displaynames(lat))
$i    Sites         : $(display_as_tuple(sublatsites(lat))) --> $(nsites(lat)) total per unit cell")
end

Base.summary(::Lattice{E,L,T}) where {E,L,T} =
    "Lattice{$E,$L,$T} : $(L)D lattice in $(E)D space"

# External API #

"""
    lattice([bravais::Bravais,] sublats::Sublat...; dim::Val{E}, type::T, names, orbitals)

Create a `Lattice{E,L,T}` with Bravais matrix `bravais` and sublattices `sublats`
converted to a common  `E`-dimensional embedding space and type `T`. To override the
embedding  dimension `E`, use keyword `dim = Val(E)`. Similarly, override type `T` with
`type = T`.

The keywords `names` can be used to rename `sublats`. Given names can be replaced to ensure
that all sublattice names are unique.

See also `LatticePresets` for built-in lattices.

# Examples
```jldoctest
julia> lattice(bravais((1, 0)), sublat((0, 0)), sublat((0, Float32(1))); dim = Val(3))
Lattice{3,1,Float32} : 1D lattice in 3D space
  Bravais vectors : ((1.0, 0.0, 0.0),)
  Sublattices     : 2
    Names         : (:A, :B)
    Sites         : (1, 1) --> 2 total per unit cell

julia> LatticePresets.honeycomb(names = (:C, :D))
Lattice{2,2,Float64} : 2D lattice in 2D space
  Bravais vectors : ((0.5, 0.866025), (-0.5, 0.866025))
  Sublattices     : 2
    Names         : (:C, :D)
    Sites         : (1, 1) --> 2 total per unit cell
```

# See also:
    LatticePresets, bravais, sublat, supercell, intracell
"""
lattice(s::Sublat, ss::Sublat...; kw...) where {E,T} = _lattice(Unitcell(s, ss...; kw...))
_lattice(u::Unitcell{E,T}) where {E,T} = Lattice(Bravais{E,T}(), u)
lattice(br::Bravais, s::Sublat, ss::Sublat...; kw...) = Lattice(br, Unitcell(s, ss...; kw...))

#######################################################################
# Supercell
#######################################################################
struct Supercell{L,L´,M<:Union{Missing,OffsetArray{Bool}},S<:SMatrix{L,L´}} # L´ is supercell dim
    matrix::S
    sites::UnitRange{Int}
    cells::CartesianIndices{L,NTuple{L,UnitRange{Int}}}
    mask::M
end

Supercell{L}(ns::Integer) where {L} =
    Supercell(SMatrix{L,L,Int,L*L}(I), 1:ns, CartesianIndices(ntuple(_->0:0, Val(L))), missing)

dim(::Supercell{L,L´}) where {L,L´} = L´

nsites(s::Supercell{L,L´,<:OffsetArray}) where {L,L´} = sum(s.mask)
nsites(s::Supercell{L,L´,Missing}) where {L,L´} = length(s.sites) * length(s.cells)

Base.CartesianIndices(s::Supercell) = s.cells

function Base.show(io::IO, s::Supercell{L,L´}) where {L,L´}
    i = get(io, :indent, "")
    print(io, i,
"Supercell{$L,$(L´)} for $(L´)D superlattice of the base $(L)D lattice
$i  Supervectors  : $(displayvectors(s.matrix))
$i  Supersites    : $(nsites(s))")
end

isinmask(s::Supercell{L,L´,<:OffsetArray}, site, dn) where {L,L´} = s.mask[site, Tuple(dn)...]
isinmask(s::Supercell{L,L´,Missing}, site, dn) where {L,L´} = true
isinmask(s::Supercell{L,L´,<:OffsetArray}, site) where {L,L´} = s.mask[site]
isinmask(s::Supercell{L,L´,Missing}, site) where {L,L´} = true

#######################################################################
# Superlattice
#######################################################################
struct Superlattice{E,L,T<:AbstractFloat,L´,S<:Supercell{L,L´},B<:Bravais{E,L,T},U<:Unitcell{E,T}} <: AbstractLattice{E,L,T}
    bravais::B
    unitcell::U
    supercell::S
end

function Base.show(io::IO, lat::Superlattice)
    i = get(io, :indent, "")
    ioindent = IOContext(io, :indent => string(i, "  "))
    print(io, i, summary(lat), "\n",
"$i  Bravais vectors : $(displayvectors(lat.bravais.matrix; digits = 6))
$i  Sublattices     : $(nsublats(lat))
$i    Names         : $(displaynames(lat))
$i    Sites         : $(display_as_tuple(sublatsites(lat))) --> $(nsites(lat)) total per unit cell\n")
    print(ioindent, lat.supercell)
end

Base.summary(::Superlattice{E,L,T,L´}) where {E,L,T,L´} =
    "Superlattice{$E,$L,$T,$L´} : $(L)D lattice in $(E)D space, filling a $(L´)D supercell"

# apply f to trues in mask. Arguments are s = sublat, oldi = old site, dn, newi = new site
function foreach_supersite(f::F, lat::Superlattice) where {F<:Function}
    newi = 0
    for s in 1:nsublats(lat), oldi in siterange(lat, s)
        for dn in CartesianIndices(lat.supercell)
            if isinmask(lat.supercell, oldi, dn)
                newi += 1
                f(s, oldi, SVector(Tuple(dn)), newi)
            end
        end
    end
    return nothing
end

#######################################################################
# AbstractLattice interface
#######################################################################
# # find SVector type that can hold all orbital amplitudes in any lattice sites
# orbitaltype(lat::AbstractLattice{E,L,T}, type::Type{Tv} = Complex{T}) where {E,L,T,Tv} =
#     _orbitaltype(SVector{1,Tv}, lat.unitcell.orbitals...)
# _orbitaltype(::Type{S}, ::NTuple{D,NameType}, os...) where {N,Tv,D,S<:SVector{N,Tv}} =
#     (M = max(N,D); _orbitaltype(SVector{M,Tv}, os...))
# _orbitaltype(t::Type{SVector{N,Tv}}) where {N,Tv} = t
# _orbitaltype(t::Type{SVector{1,Tv}}) where {Tv} = Tv

# # find SMatrix type that can hold all matrix elements between lattice sites
# blocktype(lat::AbstractLattice{E,L,T}, type::Type{Tv} = Complex{T}) where {E,L,T,Tv} =
#     _blocktype(orbitaltype(lat, Tv))
# _blocktype(::Type{S}) where {N,Tv,S<:SVector{N,Tv}} = SMatrix{N,N,Tv,N*N}
# _blocktype(::Type{S}) where {S<:Number} = S

# numorbitals(lat::AbstractLattice) = length.(lat.unitcell.orbitals)

numbertype(::AbstractLattice{E,L,T}) where {E,L,T} = T

sublat(lat::AbstractLattice, siteidx) = Int(findlast(o -> o < siteidx, lat.unitcell.offsets))

siterange(lat::AbstractLattice, sublat) = (1+lat.unitcell.offsets[sublat]):lat.unitcell.offsets[sublat+1]

offsets(lat) = lat.unitcell.offsets

# flatdim(lat::AbstractLattice) = sum(flatdims(lat))
# flatdims(lat::AbstractLattice) = sublatsites(lat) .* numorbitals(lat)

# function flatoffsets(lat::AbstractLattice)
#     v = append!([0], flatdims(lat))
#     return cumsum!(v, v)
# end

sublatsites(lat::AbstractLattice) = diff(lat.unitcell.offsets)

nsites(lat::AbstractLattice) = length(lat.unitcell.sites)
nsites(lat::AbstractLattice, sublat) = sublatsites(lat)[sublat]

nsublats(lat::AbstractLattice) = length(lat.unitcell.names)

issuperlattice(lat::Lattice) = false
issuperlattice(lat::Superlattice) = true

# External API #

"""
    sites(lat[, sublat::Int])
    lat |> sites
    lat |> sites(s::Int)

Extract the positions of all sites in a lattice, or in a specific sublattice
"""
sites(lat::AbstractLattice) = lat.unitcell.sites
sites(lat::AbstractLattice, s::Int) = view(lat.unitcell.sites, siterange(lat, s))
sites() = lat -> sites(lat)
sites(s::Int) = lat -> sites(lat, s)

# sublatindex(lat::AbstractLattice, name::NameType) = findfirst(s -> (s.name == name), lat.sublats)
# sublatindex(lat::AbstractLattice, i::Integer) = Int(i)

#######################################################################
# supercell
#######################################################################
const TOOMANYITERS = 10^8

"""
    supercell(lat::Lattice{E,L}, v::NTuple{L,Integer}...; region = missing)
    supercell(lat::Lattice{E,L}, sc::SMatrix{L,L´,Int}; region = missing)

Generates a `Superlattice` from an `L`-dimensional lattice `lat` with Bravais vectors
`br´= br * sc`, where `sc::SMatrix{L,L´,Int}` is the integer supercell matrix with the `L´`
vectors `v`s as columns. If no `v` are given, the superlattice will be bounded.

Only sites at position `r` such that `region(r) == true` will be included in the supercell.
If `region` is missing, a Bravais unit cell perpendicular to the `v` axes will be selected
for the `L-L´` non-periodic directions.

    supercell(lattice::Lattice{E,L}, factor::Integer; region = missing)

Calls `supercell` with a uniformly scaled `sc = SMatrix{L,L}(factor * I)`

    supercell(lattice::Lattice{E,L}, factors::Integer...; region = missing)

Calls `supercell` with different scaling along each Bravais vector (diagonal supercell
with factors along the diagonal)

    lat |> supercell(v...; kw...)

Functional syntax, equivalent to `supercell(lat, v...; kw...)

    supercell(h::Hamiltonian, v...; kw...)

Promotes the `Lattice` of `h` to a `Superlattice` without changing the Hamiltonian itself,
which always refers to the unitcell of the lattice.

# Examples
```jldoctest
julia> supercell(LatticePresets.honeycomb(), region = RegionPresets.circle(300))
Superlattice{2,2,Float64,0} : 2D lattice in 2D space, filling a 0D supercell
  Bravais vectors : ((0.5, 0.866025), (-0.5, 0.866025))
  Sublattices     : 2
    Names         : (:A, :B)
    Orbitals      : ((:noname,), (:noname,))
    Sites         : (1, 1) --> 2 total per unit cell
  Supercell{2,0} for 0D superlattice of the base 2D lattice
    Supervectors  : ()
    Supersites    : 652966

julia> supercell(LatticePresets.triangular(), (1,1), (1, -1))
Superlattice{2,2,Float64,2} : 2D lattice in 2D space, filling a 2D supercell
  Bravais vectors : ((0.5, 0.866025), (-0.5, 0.866025))
  Sublattices     : 1
    Names         : (:A)
    Orbitals      : ((:noname,),)
    Sites         : (1) --> 1 total per unit cell
  Supercell{2,2} for 2D superlattice of the base 2D lattice
    Supervectors  : ((1, 1), (1, -1))
    Supersites    : 2

julia> LatticePresets.square() |> supercell(3)
Superlattice{2,2,Float64,2} : 2D lattice in 2D space, filling a 2D supercell
  Bravais vectors : ((1.0, 0.0), (0.0, 1.0))
  Sublattices     : 1
    Names         : (:A)
    Orbitals      : ((:noname,),)
    Sites         : (1) --> 1 total per unit cell
  Supercell{2,2} for 2D superlattice of the base 2D lattice
    Supervectors  : ((3, 0), (0, 3))
    Supersites    : 9
```

# See also:
    unitcell
"""
supercell(v::Union{SMatrix,Tuple,SVector,Integer}...; kw...) = lat -> supercell(lat, v...; kw...)
supercell(lat::AbstractLattice{E,L}; kw...) where {E,L} = supercell(lat, SMatrix{L,0,Int}(); kw...)
supercell(lat::AbstractLattice{E,L}, factor::Integer; kw...) where {E,L} =
    supercell(lat, SMatrix{L,L,Int}(factor * I); kw...)
supercell(lat::AbstractLattice{E,L}, factors::Vararg{Integer,L}; kw...) where {E,L} =
    supercell(lat, SMatrix{L,L,Int}(Diagonal(SVector(factors))); kw...)
supercell(lat::AbstractLattice{E,L}, vecs::NTuple{L,Int}...; kw...) where {E,L} =
    supercell(lat, toSMatrix(Int, vecs...); kw...)
function supercell(lat::AbstractLattice{E,L}, scmatrix::SMatrix{L,L´,Int}; region = missing) where {E,L,L´}
    brmatrix = lat.bravais.matrix
    regionfunc = region === missing ? ribbonfunc(brmatrix, scmatrix) : region
    in_supercell_func = is_perp_dir(scmatrix)
    cells = supercell_cells(lat, regionfunc, in_supercell_func)
    ns = nsites(lat)
    mask = OffsetArray(BitArray(undef, ns, size(cells)...), 1:ns, cells.indices...)
    @inbounds for dn in cells
        dntup = Tuple(dn)
        dnvec = SVector(dntup)
        in_supercell = in_supercell_func(dnvec)
        in_supercell || (mask[:, dntup...] .= false; continue)
        r0 = brmatrix * dnvec
        for (i, site) in enumerate(lat.unitcell.sites)
            r = site + r0
            mask[i, dntup...] = in_supercell && regionfunc(r)
        end
    end
    supercell = Supercell(scmatrix, 1:ns, cells, mask)
    return Superlattice(lat.bravais, lat.unitcell, supercell)
end

# This is true whenever old ndist is perpendicular to new lattice
is_perp_dir(supercell) = let invs = pinvmultiple(supercell); dn -> iszero(new_dn(dn, invs)); end

new_dn(oldndist, (pinvs, n)) = fld.(pinvs * oldndist, n)
new_dn(oldndist, ::Tuple{<:SMatrix{0,0},Int}) = SVector{0,Int}()

wrap_dn(olddn::SVector, newdn::SVector, supercell::SMatrix) = olddn - supercell * newdn

function ribbonfunc(bravais::SMatrix{E,L,T}, supercell::SMatrix{L,L´}) where {E,L,T,L´}
    L <= L´ && return r -> true
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
    return regionfunc
end

function supercell_cells(lat::Lattice{E,L}, regionfunc, in_supercell_func) where {E,L}
    bravais = lat.bravais.matrix
    iter = BoxIterator(zero(SVector{L,Int}))
    foundfirst = false
    counter = 0
    for dn in iter
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
    return c
end

#######################################################################
# unitcell
#######################################################################
"""
    unitcell(lat::Lattice{E,L}, v::NTuple{L,Integer}...; region = missing)
    unitcell(lat::Lattice{E,L}, uc::SMatrix{L,L´,Int}; region = missing)

Generates a `Lattice` from an `L`-dimensional lattice `lat` and a larger unit cell, such
that its Bravais vectors are `br´= br * uc`. Here `uc::SMatrix{L,L´,Int}` is the integer
unitcell matrix, with the `L´` vectors `v`s as columns. If no `v` are given, the new lattice
will be bounded.

Only sites at position `r` such that `region(r) == true` will be included in the new
unitcell. If `region` is missing, a Bravais unitcell perpendicular to the `v` axes will be
selected for the `L-L´` non-periodic directions.

    unitcell(lattice::Lattice{E,L}, factor::Integer; region = missing)

Calls `unitcell` with a uniformly scaled `uc = SMatrix{L,L}(factor * I)`

    unitcell(lattice::Lattice{E,L}, factors::Integer...; region = missing)

Calls `unitcell` with different scaling along each Bravais vector (diagonal supercell
with factors along the diagonal)

    lattice |> unitcell(v...; kw...)

Functional syntax, equivalent to `unitcell(lattice, v...; kw...)

    unitcell(slat::Superlattice)

Convert Superlattice `slat` into a lattice with its unit cell matching `slat`'s supercell.

    unitcell(h::Hamiltonian, v...; kw...)

Transforms the `Lattice` of `h` to have a larger unitcell, and expanding the Hamiltonian
accordingly.

# Examples
```jldoctest
julia> unitcell(LatticePresets.honeycomb(), region = RegionPresets.circle(300))
Lattice{2,0,Float64} : 0D lattice in 2D space
  Bravais vectors : ()
  Sublattices     : 2
    Names         : (:A, :B)
    Orbitals      : ((:noname,), (:noname,))
    Sites         : (326483, 326483) --> 652966 total per unit cell

julia> unitcell(LatticePresets.triangular(), (1,1), (1, -1))
Lattice{2,2,Float64} : 2D lattice in 2D space
  Bravais vectors : ((0.0, 1.732051), (1.0, 0.0))
  Sublattices     : 1
    Names         : (:A)
    Orbitals      : ((:noname,),)
    Sites         : (2) --> 2 total per unit cell

julia> LatticePresets.square() |> unitcell(3)
Lattice{2,2,Float64} : 2D lattice in 2D space
  Bravais vectors : ((3.0, 0.0), (0.0, 3.0))
  Sublattices     : 1
    Names         : (:A)
    Orbitals      : ((:noname,),)
    Sites         : (9) --> 9 total per unit cell

julia> supercell(LatticePresets.square(), 3) |> unitcell
Lattice{2,2,Float64} : 2D lattice in 2D space
  Bravais vectors : ((3.0, 0.0), (0.0, 3.0))
  Sublattices     : 1
    Names         : (:A)
    Orbitals      : ((:noname,),)
    Sites         : (9) --> 9 total per unit cell
```

# See also:
    supercell
"""
unitcell(v::Union{SMatrix,Tuple,SVector,Integer}...; kw...) = lat -> unitcell(lat, v...; kw...)
unitcell(lat::Lattice, args...; kw...) = unitcell(supercell(lat, args...; kw...))

function unitcell(lat::Superlattice)
    newoffsets = supercell_offsets(lat)
    newsites = supercell_sites(lat)
    unitcell = Unitcell(newsites, lat.unitcell.names, newoffsets)
    bravais = lat.bravais * lat.supercell.matrix
    return Lattice(bravais, unitcell)
end

function supercell_offsets(lat::Superlattice)
    sitecounts = zeros(Int, nsublats(lat) + 1)
    foreach_supersite((s, oldi, dn, newi) -> sitecounts[s + 1] += 1, lat)
    newoffsets = cumsum!(sitecounts, sitecounts)
    return newoffsets
end

function supercell_sites(lat::Superlattice)
    newsites = similar(lat.unitcell.sites, nsites(lat.supercell))
    oldsites = lat.unitcell.sites
    bravais = lat.bravais.matrix
    foreach_supersite((s, oldi, dn, newi) -> newsites[newi] = bravais * dn + oldsites[oldi], lat)
    return newsites
end