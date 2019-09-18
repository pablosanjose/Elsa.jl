#######################################################################
# Sublattice (Sublat) : a group of identical sites (e.g. same orbitals)
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
# Supercell
#######################################################################
struct Supercell{L,O,LP,LO}
    openbravais::SMatrix{L,O,Int,LO}
    cellmask::OffsetArray{Bool,LP,BitArray{LP}}
end

Supercell{L,O}(nsites::Integer,
               ranges::NTuple{L,AbstractRange} = ntuple(_ -> 0:0, Val(L))) where {L,O} =
    Supercell(SMatrix{L,O,Int,L*O}(I),
              OffsetArray(trues(1:nsites, length.(ranges)...), 1:nsites, ranges...))

nopenboundaries(::Supercell{L,O}) where {L,O} = O

nsupercellsites(s::Supercell) = sum(s.cellmask)

function scale(s::Supercell, naxes)
    siterange = first(axes(s.cellmask))
    newmask = similar(s.cellmask, siterange, naxes...)
    bbox = boundingbox(s)
    for c in CartesianIndices(newmask)
        cd, _ = wrap(tail(Tuple(c)), bbox)
        newmask[c] = s.cellmask[first(Tuple(c)), cd...]
    end
    return Supercell(s.openbravais, newmask)
end

boundingbox(s::Supercell) = extrema.(tail(axes(s.cellmask)))

# @inline wrap(i::CartesianIndex, bbox) = wrap(Tuple(i), bbox)
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
struct Lattice{E,L,T<:AbstractFloat,B<:Bravais{E,L,T},S<:Tuple{Vararg{Sublat{E,T}}},D<:Supercell{L}}
    bravais::B
    sublats::S
    supercell::D
    offsets::Vector{Int}
end
Lattice(bravais::Bravais{E2,L}, sublats::Tuple{Vararg{Sublat{E,T}}},
        supercell = Supercell{L,L}(sum(nsites, sublats))) where {E,L,T,E2} =
    Lattice(convert(Bravais{E,L,T}, bravais), sublats, supercell, offsets(sublats))

# find SVector type that can hold all orbital amplitudes in any lattice sites
orbitaltype(lat::Lattice{E,L,T}, type::Type{Tv} = Complex{T}) where {E,L,T,Tv} =
    _orbitaltype(SVector{1,Tv}, lat.sublats...)
_orbitaltype(::Type{S}, s::Sublat{E,T,D}, ss...) where {N,Tv,E,T,D,S<:SVector{N,Tv}} =
    (M = max(N,D); _orbitaltype(SVector{M,Tv}, ss...))
_orbitaltype(t) = t

# find SMatrix type that can hold all matrix elements between lattice sites
blocktype(lat::Lattice{E,L,T}, type::Type{Tv} = Complex{T}) where {E,L,T,Tv} =
    _blocktype(orbitaltype(lat, Tv))
_blocktype(::Type{S}) where {N,Tv,S<:SVector{N,Tv}} = SMatrix{N,N,Tv,N*N}

sublat(lat::Lattice, siteidx) = findlast(o -> o < siteidx, lat.offsets)

displaynames(l::Lattice) = string("(", join(displayname.(l.sublats), ", "), ")")
displayorbitals(l::Lattice) = string("(", join(displayorbitals.(l.sublats), ", "), ")")
displayvectors(lat::Lattice) = displayvectors(lat.bravais.matrix; digits = 6)

Base.show(io::IO, lat::Lattice{E,L,T}) where {E,L,T} = print(io,
"Lattice{$E,$L,$T} : $(L)D lattice in $(E)D space
  Bravais vectors : $(displayvectors(lat))
  Sublattices     : $(nsublats(lat))
    Names         : $(displaynames(lat))
    Orbitals      : $(displayorbitals(lat))
    Sites         : $(nsites.(lat.sublats)) --> $(nsites(lat)) total per unit cell
  Supercell
    Dimensions    : $(nopenboundaries(lat.supercell))
    Total sites   : $(nsupercellsites(lat.supercell))")

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
lattice(sublats::Sublat...; kw...) = _lattice(promote(sublats...); kw...)
lattice(br::Bravais, sublats::Sublat...; kw...) = _lattice(br, promote(sublats...); kw...)

function transform!(lat::Lattice, f::Function; sublats = eachindex(lat.sublats))
    for s in sublats
        sind = sublatindex(lat, s)
        transform!(lat.sublats[sind], f)
    end
    br = transform(lat.bravais, f)
    return Lattice(br, lat.sublats, lat.supercell)
end

transform(lat::Lattice, f; kw...) = transform!(deepcopy(lat), f; kw...)
scale(lat::Lattice, s) =
    Lattice(lat.bravais, copy.(lat.sublats), scale(lat.supercell, s), copy(lat.offsets))

# Auxiliary #

_lattice(sublats::NTuple{N,Sublat{E,T}}; kw...) where {E,T,N} = _lattice(Bravais{E,T}(), sublats; kw...)
function _lattice(bravais::Bravais{E,L,T1}, sublats::NTuple{N,Sublat{E,T2}};
                 dim::Val{E2} = Val(E), type::Type{T} = promote_type(T1,T2),
                 names::NTuple{N} = (s->s.name).(sublats),
                 orbitals::NTuple{N,Tuple} = (s->s.orbitals).(sublats)) where {N,T,E,L,T1,T2,E2}
    allnames = NameType[:_]
    vecnames = collect(names)
    for i in eachindex(vecnames)
        vecnames[i] in allnames && (vecnames[i] = uniquename(allnames, vecnames[i], i))
        push!(allnames, vecnames[i])
    end
    names = ntuple(n -> vecnames[n], Val(N))
    actualsublats = Sublat{E2,float(T)}.(sublats, nametype.(names), map.(nametype, orbitals))
    return Lattice(bravais, actualsublats)
end

function uniquename(allnames, name, i)
    newname = nametype(i)
    return newname in allnames ? uniquename(allnames, name, i + 1) : newname
end

nsites(lat::Lattice) = isempty(lat.sublats) ? 0 : sum(nsites, lat.sublats)
nsublats(lat) = length(lat.sublats)

sublatindex(lat::Lattice, name::NameType) = findfirst(s -> (s.name == name), lat.sublats)
sublatindex(lat::Lattice, i::Integer) = Int(i)

function offsets(sublats)
    ns = [nsites.(sublats)...]
    tot = 0
    @inbounds for (i, n) in enumerate(ns)
        ns[i] = tot
        tot += n
    end
    return ns
end

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