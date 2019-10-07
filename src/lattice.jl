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
    semibounded::SVector{L,Bool}
end

Bravais{E,T}() where {E,T} = Bravais(SMatrix{E,0,T,0}(), ())

displayvectors(br::Bravais) = displayvectors(br.matrix)

Base.show(io::IO, b::Bravais{E,L,T}) where {E,L,T} = print(io,
"Bravais{$E,$L,$T} : set of $L Bravais vectors in $(E)D space.
  Vectors     : $(displayvectors(b))
  Matrix      : $(b.matrix),
  Semibounded : $(iszero(b.semibounded) ? "none" : findall(b.semibounded))")

# External API #

"""
    bravais(vecs...; semibounded = false)
    bravais(matrix; semibounded = false)

Create a `Bravais{E,L}` that adds `L` Bravais vectors `vecs` in `E` dimensional space,
alternatively given as the columns of matrix `mat`. For higher instantiation efficiency
enter `vecs` as `Tuple`s or `SVector`s and `mat` as `SMatrix`.

To create semibounded lattices along some or all Bravais vectors, use `semibounded`. A
`semibounded = true` makes all axes semibounded. A `semibounded = (axes::Int...)` indicates
the indices of axes to be made semibounded. A tuple of Booleans, one per Bravais axes,
`semibounded = issemi::NTuple{L,Bool}` is also allowed. Note that semibounded lattices
always extend toward positive multiples of Bravais vectors. To invert the direction, invert
the vectors.

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

# See also:
    semibounded
"""
bravais(vs::Union{Tuple, AbstractVector}...; semibounded = false, kw...) =
    (s = toSMatrix(vs...); Bravais(s, sanitize_semibounded(semibounded, s)))

bravais(lat::AbstractLattice) = lat.bravais.matrix

sanitize_semibounded(sb::Bool, ::SMatrix{E,L}) where {E,L} =
    SVector(filltuple(sb, Val(L)))
sanitize_semibounded(sb::Int, s::SMatrix{E,L}) where {E,L} =
    sanitize_semibounded((sb,), s)
sanitize_semibounded(sb::NTuple{L,Bool}, ::SMatrix{E,L}) where {E,L} = SVector(sb)
sanitize_semibounded(sb, ::SMatrix{E,L}) where {E,L} =
    SVector(ntuple(i -> i in sb, Val(L)))

transform(b::Bravais{E,0}, f::F) where {E,F<:Function} = b

function transform(b::Bravais{E,L,T}, f::F) where {E,L,T,F<:Function}
    svecs = let z = zero(SVector{E,T})
        ntuple(i -> f(b.matrix[:, i]) - f(z), Val(L))
    end
    matrix = hcat(svecs...)
    return Bravais(matrix, b.semibounded)
end

Base.:*(factor::Number, b::Bravais) = Bravais(factor * b.matrix, b.semibounded)
Base.:*(b::Bravais, factor::Number) = Bravais(b.matrix * factor, b.semibounded)

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
    names = [s.name for s in sublats], kw...) where {N,E,E2,T,T2}
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
    hassemi = any(lat.bravais.semibounded)
    print(io, i, summary(lat), "\n",
"$i  Bravais vectors : $(displayvectors(lat.bravais.matrix; digits = 6))", hassemi ? "
$i    Semibounded   : $(display_as_tuple(findall(lat.bravais.semibounded)))" : "", "
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

julia> LatticePresets.honeycomb(semibounded = 1, names = (:C, :D))
Lattice{2,2,Float64} : 2D lattice in 2D space
  Bravais vectors : ((0.5, 0.866025), (-0.5, 0.866025))
    Semibounded   : (1)
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
    semibounded::SVector{L´,Bool}
end

# Supercell{L}(ns::Integer) where {L} =
#     Supercell(SMatrix{L,L,Int,L*L}(I), 1:ns, CartesianIndices(ntuple(_->0:0, Val(L))),
#               missing, filltuple(false, Val(L)))

dim(::Supercell{L,L´}) where {L,L´} = L´

nsites(s::Supercell{L,L´,<:OffsetArray}) where {L,L´} = sum(s.mask)
nsites(s::Supercell{L,L´,Missing}) where {L,L´} = length(s.sites) * length(s.cells)

Base.CartesianIndices(s::Supercell) = s.cells

function Base.show(io::IO, s::Supercell{L,L´}) where {L,L´}
    hassemi = any(s.semibounded)
    i = get(io, :indent, "")
    print(io, i,
"Supercell{$L,$(L´)} for $(L´)D superlattice of the base $(L)D lattice
$i  Supervectors  : $(displayvectors(s.matrix))
$i  Supersites    : $(nsites(s))", hassemi ? "
$i  Semibounded   : $(display_as_tuple(findall(s.semibounded)))" : "")
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
    hassemi = any(lat.bravais.semibounded)
    ioindent = IOContext(io, :indent => string(i, "  "))
    print(io, i, summary(lat), "\n",
"$i  Bravais vectors : $(displayvectors(lat.bravais.matrix; digits = 6))", hassemi ? "
$i    Semibounded   : $(display_as_tuple(findall(lat.bravais.semibounded)))" : "", "
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
    supercell(lat::AbstractLattice{E,L}, v::NTuple{L,Integer}...; region = missing)
    supercell(lat::AbstractLattice{E,L}, sc::SMatrix{L,L´,Int}; region = missing)

Generates a `Superlattice` from an `L`-dimensional lattice `lat` with Bravais vectors
`br´= br * sc`, where `sc::SMatrix{L,L´,Int}` is the integer supercell matrix with the `L´`
vectors `v`s as columns. If no `v` are given, the superlattice will be bounded. If `lat` has
semibounded axes, these cannot be mixed with any other axes (they can only be removed, kept
intact or scaled by a factor, see below).

Only sites at position `r` such that `region(r) == true` will be included in the supercell.
If `region` is missing, a Bravais unit cell perpendicular to the `v` axes will be selected
for the `L-L´` non-periodic directions.

    supercell(lattice::AbstractLattice{E,L}, factor::Integer; region = missing)

Calls `supercell` with a uniformly scaled `sc = SMatrix{L,L}(factor * I)`

    supercell(lattice::AbstractLattice, factors::Integer...; region = missing)

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

supercell(lat::AbstractLattice{E,L}, factors::Vararg{<:Integer,L}; kw...) where {E,L} =
    _supercell(lat, factors...)
supercell(lat::AbstractLattice{E,L}, factors::Vararg{<:Integer,L´}; kw...) where {E,L,L´} =
    throw(ArgumentError("Provide either a single scaling factor or one for each of the $L lattice dimensions"))
supercell(lat::AbstractLattice{E,L}, factor::Integer; kw...) where {E,L} =
    _supercell(lat, ntuple(_ -> factor, Val(L))...)
supercell(lat::AbstractLattice, vecs::NTuple{L,Integer}...; region = missing) where {L} =
    _supercell(lat, toSMatrix(Int, vecs...), region)
supercell(lat::AbstractLattice, s::SMatrix; region = missing) = _supercell(lat, s, region)

function _supercell(lat::AbstractLattice{E,L}, factors::Vararg{Integer,L}) where {E,L,L´}
    scmatrix = SMatrix{L,L,Int}(Diagonal(SVector(factors)))
    sites = 1:nsites(lat)
    cells = CartesianIndices((i -> 0 : i - 1).(factors))
    mask = missing
    semibounded = supercell_semibounded(lat, scmatrix)
    supercell = Supercell(scmatrix, sites, cells, mask, semibounded)
    return Superlattice(lat.bravais, lat.unitcell, supercell)
end

function _supercell(lat::AbstractLattice{E,L}, scmatrix::SMatrix{L,L´,Int}, region) where {E,L,L´}
    semibounded = supercell_semibounded(lat, scmatrix)
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
    supercell = Supercell(scmatrix, 1:ns, cells, all(mask) ? missing : mask, semibounded)
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

supercell_semibounded(lat::Lattice, s::SMatrix) = supercell_semibounded(lat.bravais, s)
function supercell_semibounded(b::Bravais, s::SMatrix{L,L´,Int}) where {L,L´}
    lsb = b.semibounded
    err = ArgumentError("Cannot mix semibounded axes in a supercell")
    ssb = ntuple(Val(L´)) do i
        vec_semi = s[lsb, i]
        vec_other = s[(!).(lsb), i]
        if iszero(vec_other)
            count(!iszero, vec_semi) != 1 && throw(err)
            return true
        else
            iszero(vec_semi) || throw(err)
            return false
        end
    end
    return SVector(ssb)
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
    bravais = Bravais(lat.bravais.matrix * lat.supercell.matrix, lat.supercell.semibounded)
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

#######################################################################
# semibounded
#######################################################################
"""
    semibounded(lat::AbstractLattice, axes)

Create an AbstractLattice like `lat` but with a specific set of semibound axes. These can be
a boolean (i.e. all or none are semibounded), a tuple of booleans (one per axis) or a
collection of axis indices to make semibounded

    lat |> semibounded(axes)

Function form equivalent to semibounded(lat, axes)

# Examples
```jldoctest
julia> semibounded(LatticePresets.honeycomb(), 2)
Lattice{2,2,Float64} : 2D lattice in 2D space
  Bravais vectors : ((0.5, 0.866025), (-0.5, 0.866025))
    Semibounded   : (2)
  Sublattices     : 2
    Names         : (:A, :B)
    Sites         : (1, 1) --> 2 total per unit cell

julia> LatticePresets.honeycomb() |> supercell(4) |> semibounded((true, false))
Superlattice{2,2,Float64,2} : 2D lattice in 2D space, filling a 2D supercell
  Bravais vectors : ((0.5, 0.866025), (-0.5, 0.866025))
  Sublattices     : 2
    Names         : (:A, :B)
    Sites         : (1, 1) --> 2 total per unit cell
  Supercell{2,2} for 2D superlattice of the base 2D lattice
    Supervectors  : ((4, 0), (0, 4))
    Supersites    : 32
    Semibounded   : (1)
```
"""
semibounded(axes) = lat -> semibounded(lat, axes)

function semibounded(lat::Lattice, axes = true)
    matrix = lat.bravais.matrix
    sb = sanitize_semibounded(axes, matrix)
    br = Bravais(matrix, sb)
    return Lattice(br, lat.unitcell)
end

function semibounded(lat::Superlattice, axes = true)
    sc = lat.supercell
    matrix = lat.bravais.matrix * sc.matrix
    sb = sanitize_semibounded(axes, matrix)
    scell = Supercell(sc.matrix, sc.sites, sc.cells, sc.mask, sb)
    return Superlattice(lat.bravais, lat.unitcell, scell)
end