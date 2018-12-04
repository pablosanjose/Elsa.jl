#######################################################################
# Sublattice (Sublat) : a group of identical sites (e.g. same orbitals)
#######################################################################
"""
    Sublat([name::Symbol = missing, ]sites...)

Create a `Sublat{T,E} <: LatticeDirective` that adds a sublattice, of
name `name`, with sites at positions `sites` in `E` dimensional space.

A type `T` for coordinates can also be specified using `Sublat{T}(...)`.
For higher efficiency write `sites` as `Tuple`s or `SVector`s.

# Examples
```jldoctest
julia> Sublat(:A, (0, 0), (1, 1), (1, -1))
Sublat{Int64,2}(:A, SArray{Tuple{2},Int64,1,2}[[0, 0], [1, 1], [1, -1]])

julia> Sublat(Float32, :A, (0, 0), (1, 1), (1, -1))
Sublat{Float32,2}(:A, StaticArrays.SArray{Tuple{2},Float32,1,2}[[0.0, 0.0], [1.0, 1.0], [1.0, -1.0]])
```
"""
struct Sublat{T,E} <: LatticeDirective
    name::Union{Symbol,Missing}
    sites::Vector{SVector{E,T}}
end

Sublat(vs...) = Sublat(missing, toSVectors(vs...))
Sublat(name::Symbol, vs::(<:Union{Tuple, AbstractVector{<:Number}})...) = Sublat(name, toSVectors(vs...))
Sublat(::Type{T}, vs...) where {T} = Sublat(missing, toSVectors(T, vs...))
Sublat(::Type{T}, name::Symbol, vs...) where {T} = Sublat(name, toSVectors(T, vs...))
Sublat{T,E}(name = missing) where {T,E} = Sublat(name, SVector{E,T}[])

nsites(s::Sublat) = length(s.sites)
# dim(s::Sublat{T,E}) where {T,E} = E

transform!(s::S, f::F) where {S<:Sublat, F<:Function} = (s.sites .= f.(s.sites); s)
flatten(ss::Sublat...) = Sublat(ss[1].name, vcat((s.sites for s in ss)...))

#######################################################################
# Bravais
#######################################################################
"""
    Bravais(vecs...)
    Bravais(mat)

Create a `Bravais{T,E,L,EL} <: LatticeDirective` that adds `L` Bravais vectors
`vecs` in `E` dimensional space, alternatively given as the columns of matrix
`mat`.

A type `T` for vector elements can also be specified using `Bravais(T, vecs...)`.
For higher efficiency write `vecs` as `Tuple`s or `SVector`s and `mat`
as `SMatrix`.

# Examples
```jldoctest
julia> Bravais((1, 2), (3, 4))
Bravais{Int64,2,2,4}([1 3; 2 4])

julia> Bravais{Float64}(@SMatrix[1 2; 3 4])
Bravais{Float64,2,2,4}([1.0 2.0; 3.0 4.0])
```
"""
struct Bravais{T,E,L,EL} <: LatticeDirective
    matrix::SMatrix{E,L,T,EL}
    (::Type{Bravais})(matrix::SMatrix{E,L,T,EL}) where {T,E,L,EL} =
        arelinearindep(matrix) ? new{T,E,L,EL}(matrix) :
            throw(DomainError("Bravais vectors $(vectorsastuples(matrix)) are not linearly independent"))
end

Bravais(vs...) = Bravais(toSMatrix(vs...))
Bravais(::Type{T}, vs...) where {T} = Bravais(toSMatrix(T, vs...))

transform(b::Bravais{T,E,0,0}, f::F) where {T,E,F<:Function} = b
function transform(b::Bravais{T,E,L,EL}, f::F) where {T,E,L,EL,F<:Function}
    svecs = let z = zero(SVector{E,T})
        ntuple(i -> f(b.matrix[:, i]) - f(z), Val(L))
    end
    matrix = hcat(svecs...)
    return Bravais(matrix)
end

arelinearindep(matrix::SMatrix{E,L}) where {E,L} = E >= L && (L == 0 || fastrank(matrix) == L)

keepaxes(br::Bravais, unwrappedaxes) = Bravais(keepcolumns(bravaismatrix(br), unwrappedaxes))

################################################################################
## Dim LatticeDirective
################################################################################
"""
    Dim(E::Int)

Create a `Dim{E} <: LatticeDirective` that specifies the dimension `E` of a lattice's
embedding space.

# Examples
```jldoctest
julia> Dim(3)
Dim{3}()
```
"""
struct Dim{E} <: LatticeDirective
end

Dim(e::Int) = Dim{e}()

################################################################################
## Dim LatticeConstant
################################################################################
"""
    LatticeConstant(c[, axis = missing])

Create a `LatticeConstant{T} <: LatticeDirective` that can be used to uniformly
rescale a lattice in space, so that the resulting lattice constant along a given
`axis` is `c`. If `axis` is `missing` or there is no such `axis` in the lattice
the axis with the maximum lattice constant is chosen.

# Examples
```jldoctest
julia> LatticeConstant(2.5)
LatticeConstant{Float64}(2.5, missing)
```
"""
struct LatticeConstant{T} <: LatticeDirective
    a::T
    axis::Union{Int,Missing}
end
LatticeConstant(a) = LatticeConstant(a, missing)

################################################################################
## Region LatticeDirective
################################################################################
"""
    Region(regionname::Symbol, args...)
    Region{E}(region::Function; seed = zero(SVector{E,Float64}), excludeaxes = (), maxsteps = typemax(Int))

Create a `Region{E,F,N} <: LatticeDirective` to fill a region in `E`-dimensional
space defined by `region(r) == true`, where function `region::F` can alternatively be
defined by a region `regionname` as `Elsa.regionpresets[regionname](args...; kw...)`.

Fill search starts at position `seed`, and takes a maximum of `maxsteps` along all lattice
Bravais vectors, excluding those specified by `excludeaxes::NTuple{N,Int}`.

# Examples
```jldoctest
julia> r = Region(:circle, 10); r.region([10,10])
false

julia> r = Region(:square, 20); r.region([10,10])
true

julia> Tuple(keys(Elsa.regionpresets))
(:ellipse, :circle, :sphere, :cuboid, :cube, :rectangle, :spheroid, :square)

julia> r = Region{3}(r -> 0<=r[1]<=1 && abs(r[2]) <= sec(r[1]); excludeaxes = (3,)); r.region((0,1,2))
true
```
"""
struct Region{E,F<:Function,N} <: LatticeDirective
    region::F
    seed::SVector{E, Float64}
    excludeaxes::NTuple{N,Int}
    maxsteps::Int
end

Region(name::Symbol, args...; kw...) = regionpresets[name](args...; kw...)

Region{E}(region::F;
    seed::Union{AbstractVector,Tuple} = zero(SVector{E,Float64}),
    excludeaxes::NTuple{N,Int} = (), maxsteps = typemax(Int)) where {E,F,N} =
        Region{E,F,N}(region, SVector(seed), excludeaxes, maxsteps)

################################################################################
##   Precision LatticeDirective
################################################################################
"""
    Precision(Type)

Create a `Precision{T} <: LatticeDirective` especifying the numeric `Type` of space
coordinates and other derived quantities.

# Examples
```jldoctest
julia> Precision(Float32)
Precision{Float32}(Float32)
```
"""
struct Precision{T<:Number} <: LatticeDirective
    t::Type{T}
end

################################################################################
##   Supercell LatticeDirective
################################################################################
"""
    Supercell(inds...)
    Supercell(matrix)
    Supercell(rescaling::Int)

Create a `Supercell{S} <: LatticeDirective` that defines a supercell in terms of a
vectors of integer indices `inds...` that relate the new lattice vectors `v` to
the original ones `v0` as `v[i] = inds[i][j] * v0[j]`. Alternatively, `inds...` can be
given as columns of `matrix` or as a uniform `rescaling` so that `mat = rescaling * I`.

For higher efficiency, write `inds...` as several `NTuple{N,Int}` or `SVectors`, and
`matrix` as an `SMatrix`. Parameter `S <: SMatrix` is the type of the `matrix` stored
internally.

# Examples
```jldoctest
julia> Supercell((1,2), (3,4))
Supercell{StaticArrays.SArray{Tuple{2,2},Int64,2,4}}([1 3; 2 4])
```
"""
struct Supercell{S} <: LatticeDirective
    matrix::S
end

Supercell(rescaling::Number) = Supercell(Int(rescaling)*I)
Supercell(vs::(<:Union{Tuple,SVector})...) = Supercell(toSMatrix(Int, vs...))
Supercell(rescalings::Vararg{Number,N}) where {N} = Supercell(SMatrix{N,N,Int}(Diagonal(SVector(rescalings))))

#######################################################################
## LinkRule LatticeDirective : directives to create links in a lattice
#######################################################################
abstract type LinkingAlgorithm end

struct SimpleLinking{F<:Function} <: LinkingAlgorithm
    isinrange::F
end
SimpleLinking(range::Number) = SimpleLinking(dr -> norm(dr) <= range + extended_eps())

struct TreeLinking <: LinkingAlgorithm
    range::Float64
    leafsize::Int
end
TreeLinking(range; leafsize = 10) = TreeLinking(abs(range), leafsize)

struct WrapLinking{T,E,L,EL,N} <: LinkingAlgorithm
    links::Links{T,E,L}
    bravais::Bravais{T,E,L,EL}
    unwrappedaxes::NTuple{N,Int}
end

struct AutomaticRangeLinking <: LinkingAlgorithm
    range::Float64
end

struct BoxIteratorLinking{T,E,L,N,EL,O<:SMatrix,C<:SMatrix} <: LinkingAlgorithm
    links::Links{T,E,L}
    iter::BoxIterator{N}
    open2old::O
    iterated2old::C
    bravais::Bravais{T,E,L,EL}
    nslist::Vector{Int}
end

"""
    LinkRule(algorithm[, sublats...]; mincells = 0, maxsteps = typemax(Int))
    LinkRule(range[, sublats...]; mincells = 0, maxsteps = typemax(Int)))

Create a `LinkRule{S,SL} <: LatticeDirective` to compute links between sites in
sublattices indicated by `sublats::SL` using `algorithm::S <: LinkingAlgorithm`.
`TreeLinking(range::Number)` and `SimpleLinking(isinrange::Function)` are available.
If a linking `range` instead an `algorithm` is given, the algorithm is chosen
automatically.

Sublattices `sublats...` to be linked are indicated by pairs of integers or sublattice
names (of type `:Symbol`). A single integer or name can be used to indicate
intra-sublattice links for that sublattice. `mincells` indicates a minimum cell search
distance, and `maxsteps` a maximum number of cells to link.

# Examples
```jldoctest
julia> lr = LinkRule(1, (2, :A), 1, (3,1)); (lr.alg.range, lr.sublats)
(1.0, ((2, :A), (1, 1), (3, 1)))

julia> LinkRule(2.0, (1, 2)) |> typeof
LinkRule{Elsa.AutomaticRangeLinking,Tuple{Tuple{Int64,Int64}}}

julia> LinkRule(TreeLinking(2.0)) |> typeof
LinkRule{TreeLinking,Missing}

julia> LinkRule(SimpleLinking(dr -> norm(dr, 1) < 2.0)) |> typeof
LinkRule{SimpleLinking{getfield(Main, Symbol("##9#10"))},Missing}
```
"""
struct LinkRule{S<:LinkingAlgorithm, SL} <: LatticeDirective
    alg::S
    sublats::SL
    mincells::Int  # minimum range to search in using BoxIterator
    maxsteps::Int
end
LinkRule(range; kw...) = LinkRule(; range = range, kw...)
LinkRule(range, sublats...; kw...) = LinkRule(; range = range, sublats = sublats, kw...)
LinkRule(; range = 10.0, sublats = missing, kw...) =
    LinkRule(AutomaticRangeLinking(abs(range)), tuplesort(to_tuples_or_missing(sublats)); kw...)
LinkRule(alg::S; kw...) where S<:LinkingAlgorithm = LinkRule(alg, missing; kw...)
LinkRule(alg::S, sublats; mincells = 0, maxsteps = typemax(Int)) where S<:LinkingAlgorithm =
    LinkRule(alg, sublats, abs(mincells), maxsteps)

#######################################################################
# Lattice : group of sublattices + Bravais vectors + links
#######################################################################
"""
    Lattice([preset, ]directives...)

Build a `Lattice{T,E,L,EL}` of `L` dimensions in `E`-dimensional embedding
spaceof and composed of `T`-typed sites. Optional `preset::Union{Preset, Symbol}`
is one of the `Elsa.latticepresets` or a `Preset(preset, args...)`.
The `directives::LatticeDirective...` apply additional build instructions in order.

# Examples
```jldoctest
julia> Lattice(Sublat(:C, (1,0), (0,1)), Sublat(:D, (0.5,0.5)), Bravais((1,0), (0,2)), Dim(3))
Lattice{Float64,3,2} : 2D lattice in 3D space with Float64 sites
    Bravais vectors : ((1.0, 0.0, 0.0), (0.0, 2.0, 0.0))
    Number of sites : 3
    Sublattice names : (:C, :D)
    Unique Links : 0

julia> Lattice(:honeycomb, LinkRule(1/√3), Region(:circle, 100))
Lattice{Float64,2,0} : 0D lattice in 2D space with Float64 sites
    Bravais vectors : ()
    Number of sites : 72562
    Sublattice names : (:A, :B)
    Unique Links : 108445

julia> Lattice(Preset(:honeycomb_bilayer, twistindices=(31,1)), Precision(Float32))
Lattice{Float32,3,2} : 2D lattice in 3D space with Float32 sites
    Bravais vectors : ((-0.0f0, 54.56189f0, 0.0f0), (-47.251984f0, 27.280945f0, 0.0f0))
    Number of sites : 11908
    Sublattice names : (:Ab, :Bb, :At, :Bt)
    Unique Links : 18171

julia> Tuple(keys(Elsa.latticepresets))
(:bcc, :graphene, :honeycomb, :cubic, :linear, :fcc, :honeycomb_bilayer, :square, :triangular)
```
"""
mutable struct Lattice{T,E,L,EL}
    sublats::Vector{Sublat{T,E}}
    bravais::Bravais{T,E,L,EL}
    links::Links{T,E,L}
end
Lattice(sublats::Vector{Sublat{T,E}}, bravais::Bravais{T,E,L}) where {T,E,L} = Lattice(sublats, bravais, dummylinks(sublats, bravais))
# Lattice(bravais::Bravais{T,E,L}) where {T,E,L} = Lattice([Sublat(zero(SVector{E,T}))], bravais)

Lattice(name::Symbol) = Lattice(Preset(name))
Lattice(name::Symbol, opts...) = lattice!(Lattice(name), opts...)
Lattice(preset::Preset) = latticepresets[preset.name](; preset.kwargs...)
Lattice(preset::Preset, opts...) = lattice!(Lattice(preset), opts...)
Lattice(opts::LatticeDirective...) = lattice!(seedlattice(Lattice{Float64,0,0,0}, opts...), opts...)

# Vararg here is necessary in v0.7-alpha (instead of ...) to make these type-unstable recursions fast
seedlattice(::Type{S}, opts::Vararg{<:LatticeDirective,N}) where {S, N} = _seedlattice(seedtype(S, opts...))
_seedlattice(::Type{Lattice{T,E,L,EL}}) where {T,E,L,EL} = Lattice(Sublat{T,E}[], Bravais(SMatrix{E,L,T,EL}(I)))
seedtype(::Type{T}, t, ts::Vararg{<:Any, N}) where {T,N} = seedtype(seedtype(T, t), ts...)
seedtype(::Type{Lattice{T,E,L,EL}}, ::Sublat{T2,E2}) where {T,E,L,EL,T2,E2} = Lattice{T,E2,L,E2*L}
seedtype(::Type{S}, ::Bravais{T2,E,L,EL}) where {T,S<:Lattice{T},T2,E,L,EL} = Lattice{T,E,L,EL}
seedtype(::Type{Lattice{T,E,L,EL}}, ::Dim{E2}) where {T,E,L,EL,E2} = Lattice{T,E2,L,E2*L}
seedtype(::Type{Lattice{T,E,L,EL}}, ::Precision{T2}) where {T,E,L,EL,T2} = Lattice{T2,E,L,EL}
seedtype(::Type{S}, ::Region, ts...) where {S<:Lattice} = S
seedtype(::Type{L}, opt) where {L<:Lattice} = L

#######################################################################
# Apply LatticeDirectives
#######################################################################
"""
    lattice!(lat::Lattice, directives::LatticeDirective...)

Apply `directives` to lattice `lat` in order, modifying it in place whenever
possible.

# Examples
```jldoctest
julia> lattice!(Lattice(:honeycomb), Sublat(:C, (0, 0)))
Lattice{Float64,2,2} : 2D lattice in 2D space with Float64 sites
    Bravais vectors : ((0.5, 0.866025), (-0.5, 0.866025))
    Number of sites : 3
    Sublattice names : (:A, :B, :C)
    Unique Links : 0
```
"""
lattice!(lat::Lattice, o1::LatticeDirective, opts...) = lattice!(_lattice!(lat, o1), opts...)

lattice!(lat::Lattice) = adjust_slinks_to_sublats!(lat)

# Ensure the size of Slink matrices in lat.links matches the number of sublats. Do it only at the end.
function adjust_slinks_to_sublats!(lat::Lattice)
    ns = nsublats(lat)
    lat.links.intralink = resizeilink(lat.links.intralink, lat)
    for (n, ilink) in enumerate(lat.links.interlinks)
        lat.links.interlinks[n] = resizeilink(ilink, lat)
    end
    return lat
end

function resizeilink(ilink::IL, lat::Lattice) where {IL<:Ilink}
    nsold = size(ilink.slinks, 1)
    ns = nsublats(lat)
    nsold == ns && return ilink
    newslinks = padrightbottom(ilink.slinks, ns, ns) # fills with dummy 0x0 Slink
    IL(ilink.ndist, newslinks)
end

function _lattice!(lat::Lattice, b::Bravais)
    lat.bravais = b
    return lat
end

function _lattice!(lat::Lattice, s::Sublat)
    push!(lat.sublats, s)
    return lat
end

_lattice!(lat::Lattice{T,E,L,EL}, d::Dim{E2}) where {T,E,L,EL,E2} = convert(Lattice{T,E2,L,E2*L}, lat)

_lattice!(lat::Lattice{T,E,L,EL}, p::Precision{T2}) where {T,E,L,EL,T2} =
    convert(Lattice{T2,E,L,EL}, lat)

_lattice!(lat::Lattice, sc::Supercell) = expand_unitcell(lat, sc)

_lattice!(lat::Lattice, lr::LinkRule) = link!(lat, lr)

function _lattice!(lat::Lattice{T,E,L}, l::LatticeConstant) where {T,E,L}
    if L == 0
        throw(DimensionMismatch("Cannot redefine the LatticeConstant of a non-periodic lattice"))
    else
        if ismissing(l.axis) || !(1 <= l.axis <= L)
            axisnorm = maximum(norm(bravaismatrix(lat)[:,i]) for i in size(lbravaismatrix(lat), 2))
        else
            axisnorm = norm(bravaismatrix(lat)[:, l.axis])
        end
        rescale = let factor = l.a / axisnorm
            r -> factor * r
        end
        transform!(lat, rescale)
    end
    return lat
end

function _lattice!(lat::Lattice{T,E}, fr::Region{E}) where {T,E}
    fill_region(lat, fr)
end

#######################################################################
# Lattice display
#######################################################################

Base.show(io::IO, lat::Lattice{T,E,L}) where {T,E,L}=
    print(io, "Lattice{$T,$E,$L} : $(L)D lattice in $(E)D space with $T sites
    Bravais vectors  : $(vectorsastuples(lat))
    Sublattice names : $((sublatnames(lat)... ,))
    Total sites      : $(nsites(lat))
    Total links      : $(nlinks(lat))
    Coordination     : $(coordination(lat))")

#######################################################################
# Lattice utilities
#######################################################################

vectorsastuples(lat::Lattice) = vectorsastuples(lat.bravais.matrix)
vectorsastuples(br::Bravais) =  vectorsastuples(br.matrix)
vectorsastuples(mat::SMatrix{E,L}) where {E,L} = ntuple(l -> round.((mat[:,l]... ,), digits = 6), Val(L))
nsites(lat::Lattice, s::Int) = s > nsublats(lat) ? 0 : nsites(lat.sublats[s])
nsites(lat::Lattice) = isempty(lat.sublats) ? 0 : sum(nsites, lat.sublats)
nsiteslist(lat::Lattice) = [nsites(sublat) for sublat in lat.sublats]
nsublats(lat::Lattice)::Int = length(lat.sublats)
sublatnames(lat::Lattice) = Union{Symbol,Missing}[slat.name for slat in lat.sublats]
nlinks(lat::Lattice) = nlinks(lat.links)
isunlinked(lat::Lattice) = nlinks(lat.links) == 0
coordination(lat::Lattice) = (2 * nlinks(lat.links.intralink) + nlinks(lat.links.interlinks))/nsites(lat)
allilinks(lat::Lattice) = allilinks(lat.links)
getilink(lat::Lattice, i::Int) = getilink(lat.links, i)
bravaismatrix(lat::Lattice) = bravaismatrix(lat.bravais)
bravaismatrix(br::Bravais) = br.matrix
sitegenerator(lat::Lattice) = (site for sl in lat.sublats for site in sl.sites)
linkgenerator_r1r2(ilink::Ilink) = ((rdr[1] -rdr[2]/2, rdr[1] + rdr[2]/2) for s in ilink.slinks for (_,rdr) in neighbors_rdr(s))
selectbravaisvectors(lat::Lattice{T, E}, bools::AbstractVector{Bool}, ::Val{L}) where {T,E,L} =
   Bravais(SMatrix{E,L,T}(lat.bravais.matrix[1:E,bools]))

# static placeholders for unlinked lattices
const dummyslinkF64 = ntuple(E -> Slink{Float64,E}(0,0), Val(4))
const dummyslinkF32 = ntuple(E -> Slink{Float32,E}(0,0), Val(4))
const dummyslinkF16 = ntuple(E -> Slink{Float16,E}(0,0), Val(4))
dummyslink(S::Type{Slink{Float64,E}}) where {E} = _dummyslink(S, dummyslinkF64)
dummyslink(S::Type{Slink{Float32,E}}) where {E} = _dummyslink(S, dummyslinkF32)
dummyslink(S::Type{Slink{Float16,E}}) where {E} = _dummyslink(S, dummyslinkF16)
@noinline _dummyslink(S::Type{Slink{T,E}}, dummytuple) where {T,E} = E > length(dummytuple) ? S(0,0) : dummytuple[E]
dummyslinks(lat::Lattice) = dummyslinks(lat.sublats)
dummyslinks(sublats::Vector{Sublat{T,E}}) where {T,E} = fill(dummyslink(Slink{T,E}), length(sublats), length(sublats))
dummyilink(ndist::SVector, sublats::Vector{<:Sublat}) = Ilink(ndist, dummyslinks(sublats))
dummylinks(lat) = dummylinks(lat.sublats, lat.bravais)
dummylinks(sublats::Vector{Sublat{T,E}}, bravais::Bravais{T,E,L}) where {T,E,L} =
    Links(dummyilink(zero(SVector{L, Int}), sublats), Ilink{T,E,L}[])

SparseMatrixBuilder(lat::Lattice{T,E,L}, s1, s2) where {T,E,L} =
    SparseMatrixBuilder(Tuple{SVector{E,T}, SVector{E,T}}, nsites(lat, s2), nsites(lat, s1), max((L + 1), coordination(lat)))

function boundingboxlat(lat::Lattice{T,E}) where {T,E}
    bmin = zero(MVector{E, T})
    bmax = zero(MVector{E, T})
    foreach(sl -> foreach(s -> _boundingboxlat!(bmin, bmax, s), sl.sites), lat.sublats)
    return (bmin, bmax)
end
@inline function _boundingboxlat!(bmin, bmax, site)
    bmin .= min.(bmin, site)
    bmax .= max.(bmax, site)
    return nothing
end

supercellmatrix(s::Supercell{<:UniformScaling}, lat::Lattice{T,E,L}) where {T,E,L} = SMatrix{L,L}(s.matrix.λ .* one(SMatrix{L,L,Int}))
supercellmatrix(s::Supercell{<:SMatrix}, lat::Lattice{T,E,L}) where {T,E,L} = s.matrix

function matchingsublats(lat::Lattice, lr::LinkRule{S,Missing}) where S
    ns = nsublats(lat)
    match = vec([i.I for i in CartesianIndices((ns, ns))])
    return match
end
matchingsublats(lat::Lattice, lr::LinkRule{S,T}) where {S,T} = _matchingsublats(sublatnames(lat), lr.sublats)
function _matchingsublats(sublatnames, lrsublats)
    match = Tuple{Int,Int}[]
    for (s1, s2) in lrsublats
        m1 = __matchingsublats(s1, sublatnames)
        m2 = __matchingsublats(s2, sublatnames)
        m1 isa Int && m2 isa Int && push!(match, tuplesort((m1, m2)))
    end
    return sort!(match)
end
__matchingsublats(s::Int, sublatnames) = s <= length(sublatnames) ? s : nothing
__matchingsublats(s, sublatnames) = findfirst(isequal(s), sublatnames)

#######################################################################
# Transform lattices
#######################################################################
"""
    transform!(lat::Lattice, f::Function)

Transform a `Lattice` `lat` in place so that any site at position `r::SVector`
is moved to position `f(r)`. Links between sites are preserved.

# Examples
```jldoctest
julia> transform!(Lattice(:cubic, LinkRule(1)), r -> 2r + SVector(0,0,r[1]))
Lattice{Float64,3,3} : 3D lattice in 3D space with Float64 sites
    Bravais vectors : ((2.0, 0.0, 1.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0))
    Number of sites : 1
    Sublattice names : (missing,)
    Unique Links : 6
```
"""
function transform!(l::L, f::F) where {L<:Lattice, F<:Function}
    for s in l.sublats
        transform!(s, f)
    end
    isunlinked(l) || transform!(l.links, f)
    l.bravais = transform(l.bravais, f)
    return l
end

"""
    transform(lat::Lattice, f::Function)

Create a new `Lattice` as a transformation of lattice `lat` so that any site
at position `r::SVector` is moved to position `f(r)`. Links between sites are
kept the same as in `lat`.

# Examples
```jldoctest
julia> transform(Lattice(:cubic, LinkRule(1)), r -> 2r + SVector(0,0,r[1]))
Lattice{Float64,3,3} : 3D lattice in 3D space with Float64 sites
    Bravais vectors : ((2.0, 0.0, 1.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0))
    Number of sites : 1
    Sublattice names : (missing,)
    Unique Links : 6
```
"""
transform(l::Lattice, f::F) where F<:Function = transform!(deepcopy(l), f)

#######################################################################
# Combine lattices
#######################################################################
"""
    combine(lats::Lattice...)

Create a new `Lattice` from a set of lattices `lats...` that include a copy
of all the different sublattices of `lats...`. All `lats` must share compatible
Bravais vectors.

# Examples
```jldoctest
julia> combine(Lattice(:honeycomb), Lattice(:triangular))
Lattice{Float64,2,2} : 2D lattice in 2D space with Float64 sites
    Bravais vectors : ((0.5, 0.866025), (-0.5, 0.866025))
    Number of sites : 3
    Sublattice names : (:A, :B, missing)
    Unique Links : 0
```
"""
function combine(lats::Lattice...)
    combine_nocopy(deepcopy.(lats)...)
end

"""
    combine_nocopy(lats::Lattice...)

Create a new `Lattice` from a set of lattices `lats...` that include all the
original sublattices of `lats...` (not a copy). All `lats` must share compatible
Bravais vectors.

See also: [`combine`](@ref)
```
"""
function combine_nocopy(lats::Lattice...)
    bravais = check_compatible_bravais(map(lat -> lat.bravais, lats))
    combined_sublats = vcat(map(lat -> lat.sublats, lats)...)
    combined_links = combine_links(lats, combined_sublats)
    return Lattice(combined_sublats, bravais, combined_links)
end

function check_compatible_bravais(bs::NTuple{N,B}) where {N,B<:Bravais}
    allsame(bs) || throw(DimensionMismatch("Cannot combine lattices with different Bravais vectors, $(vectorsastuples.(bs))"))
    return(first(bs))
end
function ==(b1::B, b2::B) where {T,E,L,B<:Bravais{T,E,L}}
    vs1 = MVector{L,SVector{E,T}}(ntuple(i -> b1.matrix[:,i], Val(L))); sort!(vs1)
    vs2 = MVector{L,SVector{E,T}}(ntuple(i -> b2.matrix[:,i], Val(L))); sort!(vs2)
    # Caution: potential problem for equal bravais modulo signs
    all(vs->isapprox(vs[1],vs[2]), zip(vs1,vs2))
end

function combine_links(lats::NTuple{N,LL}, combined_sublats) where {N,T,E,L,LL<:Lattice{T,E,L}}
    intralink = combine_ilinks(map(l -> l.links.intralink, lats), combined_sublats)
    interlinks = Ilink{T,E,L}[]
    ndists = SVector{L,Int}[]
    for lat in lats, is in lat.links.interlinks
        if !(is.ndist in ndists)
            push!(ndists, is.ndist)
        end
    end
    ilinks = Ilink{T,E,L}[]
    for ndist in ndists
        resize!(ilinks, 0)
        for lat in lats
            push!(ilinks, getilink(lat, ndist))
        end
        push!(interlinks, combine_ilinks(ilinks, combined_sublats))
    end
    return Links(intralink, interlinks)
end

function combine_ilinks(is, combined_sublats)
    allsame(i.ndist for i in is) || throw(DimensionMismatch("Cannot combine Ilinks with different ndist"))
    ilink = dummyilink(first(is).ndist, combined_sublats)
    slinkmatrices = map(i -> i.slinks, is)
    filldiag!(ilink.slinks, slinkmatrices)
    return ilink
end

function getilink(lat::Lattice, ndist)
    if iszero(ndist)
        return lat.links.intralink
    else
        index = findfirst(i -> i.ndist == ndist, lat.links.interlinks)
        if index === nothing
            return dummyilink(ndist, lat.sublats)
        else
            return lat.links.interlinks[index]
        end
    end
end

#######################################################################
# Wrap lattice
#######################################################################

function wrap(lat::Lattice{T,E,L}; exceptaxes::NTuple{N,Int} = ()) where {T,E,L,N}
    newsublats = deepcopy(lat.sublats)
    newbravais = keepaxes(lat.bravais, exceptaxes)
    newlat = lattice!(Lattice(newsublats, newbravais),
                      LinkRule(WrapLinking(lat.links, lat.bravais, exceptaxes)))
    return newlat
end

#######################################################################
# Merge sublattices
#######################################################################
"""
    mergesublats(lat::Lattice[, newsublats::NTuple{N,Int}])

Create a new `Lattice` by merging the sublattices of `lat` as indicated by
`newsublats = (n1, n2, n3...)`, so that old sublattice `i` becomes new
sublattice `ni`. `newsublats` length `N` must match the number of `lat`
sublattices, and defaults to `ni = 1` (merge all sublattices) if absent.

# Examples
```jldoctest
julia> mergesublats(Lattice(Preset(:honeycomb_bilayer, twistindex = 2)), (2,1,1,1))
Lattice{Float64,3,2} : 2D lattice in 3D space with Float64 sites
    Bravais vectors  : ((1.0, 1.732051, 0.0), (-1.0, 1.732051, 0.0))
    Sublattice names : (:Bb, :Ab)
    Total sites      : 16
    Total links      : 32
    Coordination     : 3.0
```
"""
mergesublats(lat::Lattice) = mergesublats(lat::Lattice, ntuple(_ -> 1, nsublats(lat)))

function mergesublats(lat::Lattice, newsublats::NTuple{N,Int}) where {T,E,L,N}
    N == nsublats(lat) || throw(DimensionMismatch("The length $N of new sublattice indices should match the number $(nsublats(lat)) of sublattices in the lattice"))
    newns = maximum(newsublats)
    oldsublatlist = [findall(isequal(s), newsublats) for s in 1:newns]
    newsublats = _mergedsublats(lat, oldsublatlist)
    newbravais = lat.bravais
    newlinks = _mergedlinks(lat, oldsublatlist, newsublats)
    return Lattice(newsublats, newbravais, newlinks)
end

function _mergedsublats(lat::Lattice{T,E}, oldsublatlist) where {T,E}
    newsublats = Sublat{T,E}[]
    for oldss in oldsublatlist
        if isempty(oldss)
            push!(newsublats, Sublat{T,E}(missing))
        else
            newsublat = Sublat{T,E}(lat.sublats[oldss[1]].name)
            push!(newsublats, newsublat)
            for olds in oldss
                append!(newsublat.sites, lat.sublats[olds].sites)
            end
        end
    end
    return newsublats
end

function _mergedlinks(lat::Lattice, oldsublatlist, newsublats)
    intralink = _mergedilink(lat.links.intralink, oldsublatlist, newsublats)
    interlinks = [_mergedilink(ilink, oldsublatlist, newsublats) for ilink in lat.links.interlinks]
    return Links(intralink, interlinks)
end

function _mergedilink(oldilink, oldsublatlist, newsublats)
    newilink = dummyilink(oldilink.ndist, newsublats)
    for (s1, oldss1) in enumerate(oldsublatlist), (s2, oldss2) in enumerate(oldsublatlist)
        if !isempty(oldss1) && !isempty(oldss2)
            # rowlengths = ntuple(_ -> length(oldss1), length(oldss2))
            # rows = Tuple((oldilink.slinks[j, i].rdr for i in oldss1, j in oldss2))
            newslink = _mergedslinks(oldilink.slinks, oldss1, oldss2)
            newilink.slinks[s2, s1] = newslink
        end
    end
    return newilink
end

function _mergedslinks(slinks::Matrix{S}, indcols, indrows) where {T, E, S<:Slink{T,E}}
    blockwidths  = [maximum((nsources(slinks[i, j]) for i in indrows)) for j in indcols]
    blockheights = [maximum((ntargets(slinks[i, j]) for j in indcols)) for i in indrows]
    totalcols = sum(blockwidths)
    totalrows = sum(blockheights)
    builder = SparseMatrixBuilder(Tuple{SVector{E,T},SVector{E,T}}, totalrows, totalcols)
    for (j, jblock) in enumerate(indcols)
        for col in 1:blockwidths[j]
            rowoffset = 0
            for (i, iblock) in enumerate(indrows)
                sl = slinks[iblock, jblock]
                if size(sl.rdr) == (0,0)
                    rowoffset += blockheights[i]
                    continue
                else
                    rows = rowvals(sl.rdr)
                    vals = nonzeros(sl.rdr)
                    for rowptr in nzrange(sl.rdr, col)
                        row = rows[rowptr]
                        val = vals[rowptr]
                        pushtocolumn!(builder, row + rowoffset, val)
                    end
                    rowoffset += blockheights[i]
                end
            end
            finalisecolumn!(builder)
        end
    end
    return Slink(sparse(builder))
end

#######################################################################
# siteclusters : find disconnected site groups in a sublattice
#######################################################################
# We have a queue of pending sites (nodes). Each time its emptied we open a new bin, which is assigned to a new cluster (binclusters[newbin] = newcluster). We add one unclassified site to the queue (one whose bin is zero, sitebins[site] == 0), and start crawling its neighbors. Unclassified neighbors are added to the current bin, and placed in the pending queue. If we find a classified neighbor belonging to a older cluster, the bin cluster is changed to that cluster.

function siteclusters(lat::Lattice, sublat::Int, onlyintra)
    isunlinked(lat) && return [Int[]]

    ns = nsites(lat.sublats[sublat])
    sitebins = fill(0, ns)  # sitebins[site] = bin
    binclusters = Int[]     # binclusters[bin] = cluster number
    pending = Int[]

    bincounter = 0
    clustercounter = 0
    neighiter = NeighborIterator(lat.links, 1, (sublat, sublat), onlyintra)
    p = Progress(ns, 1, "Clustering nodes: ")
    while !isempty(pending) || any(iszero, sitebins)
        if isempty(pending)   # new cluster
            seed = findfirst(iszero, sitebins)
            bincounter += 1
            clustercounter = isempty(binclusters) ? 1 : maximum(binclusters) + 1
            sitebins[seed] = bincounter; next!(p)
            push!(binclusters, clustercounter)
            push!(pending, seed)
        end
        src = pop!(pending)
        for neigh in neighbors!(neighiter, src)
            if sitebins[neigh] == 0   # unclassified neighbor
                push!(pending, neigh)
                sitebins[neigh] = bincounter; next!(p)
            else
                clustercounter = min(clustercounter, binclusters[sitebins[neigh]])
                binclusters[bincounter] = clustercounter
            end
        end
    end
    clusters = Vector{Int}[Int[] for _ in 1:maximum(binclusters)]
    for i in 1:ns
        push!(clusters[binclusters[sitebins[i]]], i)
    end
    return clusters
end

################################################################################
## expand_unitcell
################################################################################

function expand_unitcell(lat::Lattice{T,E,L,EL}, supercell::Supercell) where {T,E,L,EL}
    if L == 0
        @warn("cannot expand a non-periodic lattice")
        return(lat)
    end
    smat = supercellmatrix(supercell, lat)
    newbravais = bravaismatrix(lat) * smat
    invscell = inv(smat)
    fillaxesbool = fill(true, L)
    seed = zero(SVector{E,T})
    isinregion =
        let invscell = invscell # avoid boxing issue #15276 (depends on compiler, but just in case)
            cell -> all(e -> - extended_eps() <= e < 1 - extended_eps(), invscell * cell)
        end
    newsublats, iter = _box_fill(Val(L), lat, isinregion, fillaxesbool, seed, missing, true)
    newlattice = Lattice(newsublats, Bravais(newbravais))
    open2old = smat
    iterated2old = one(SMatrix{L,L,Int})
    return isunlinked(lat) ? newlattice :
        link!(newlattice, LinkRule(BoxIteratorLinking(lat.links, iter, open2old,
            iterated2old, lat.bravais, nsiteslist(lat)); mincells = cellrange(lat.links)))
end

################################################################################
# fill_region() : fill a region with a lattice
################################################################################

function fill_region(lat::Lattice{T,E,L}, fr::Region{E,F,N}) where {T,E,L,F,N} #N is number of excludeaxes
    L == 0 && error("Non-periodic lattice cannot be used for region fill")
    fillaxesbool = [!any(i .== fr.excludeaxes) for i=1:L]
    filldims = L - N
    filldims == 0 && error("Need at least one lattice vector to fill region")
    any(fr.region(fr.seed + site) for site in sitegenerator(lat)) || error("Unit cell centered at seed position does not contain any site in region")

    newsublats, iter = _box_fill(Val(filldims), lat, fr.region, fillaxesbool, fr.seed, fr.maxsteps, false)

    closeaxesbool = SVector{L}(fillaxesbool)
    openaxesbool = (!).(closeaxesbool)
    newbravais = selectbravaisvectors(lat, openaxesbool, Val(N))
    newlattice = Lattice(newsublats, newbravais)
    open2old = nmatrix(openaxesbool, Val(N))
    iterated2old = nmatrix(closeaxesbool, Val(filldims))
    return isunlinked(lat) ? newlattice :
        link!(newlattice, LinkRule(BoxIteratorLinking(lat.links, iter, open2old, iterated2old,
                                    lat.bravais, nsiteslist(lat)); mincells = cellrange(lat.links)))
end

function _box_fill(::Val{N}, lat::Lattice{T,E,L}, isinregion::F, fillaxesbool, seed0, maxsteps, usecellaspos) where {N,T,E,L,F}
    seed = convert(SVector{E,T}, seed0)
    fillvectors = SMatrix{E, N}(bravaismatrix(lat)[1:E, fillaxesbool])
    numsublats = nsublats(lat)
    nregisters = ifelse(isunlinked(lat), 0, numsublats)
    nsitesub = Int[nsites(sl) for sl in lat.sublats]

    pos_sites = Vector{SVector{E,T}}[SVector{E,T}[seed + site for site in slat.sites] for slat in lat.sublats]
    newsublats = Sublat{T,E}[Sublat(sl.name, Vector{SVector{E,T}}()) for sl in lat.sublats]
    zeroseed = ntuple(_->0, Val(N))
    iter = BoxIterator(zeroseed, maxiterations = maxsteps, nregisters = nregisters)

    for cell in iter
        inregion = false
        cellpos = fillvectors * SVector(cell)
        for sl in 1:numsublats, siten in 1:nsitesub[sl]
            sitepos = cellpos + pos_sites[sl][siten]
            checkpos = usecellaspos ? SVector(cell) : sitepos
            if isinregion(checkpos)
                inregion = true
                push!(newsublats[sl].sites, sitepos)
                nregisters == 0 || registersite!(iter, cell, sl, siten)
            end
        end
        inregion && acceptcell!(iter, cell)
    end

    return newsublats, iter
end

# converts ndist in newlattice (or in fillcells) to ndist in oldlattice
nmatrix(axesbool::SVector{L,Bool}, ::Val{N}) where {L,N} =
    SMatrix{L,N,Int}(one(SMatrix{L,L,Int})[1:L, axesbool])
cellrange(links::Links) = isempty(links.interlinks) ? 0 : maximum(max(abs.(ilink.ndist)...) for ilink in links.interlinks)

#######################################################################
# Links interface
#######################################################################
# Linking rules for an Matrix{Slink}_{s2 j, s1 i} at ndist, enconding
# the link (s1, r[i]) -> (s2, r[j]) + ndist
# Linking rules are given by isvalidlink functions. With the restrictive
# intralink choice i < j we can append new sites without reordering the
# intralink slink lists (it's lower-triangular sparse)

isvalidlink(isinter::Bool, (s1, s2)) = isinter || s1 <= s2
isvalidlink(isinter::Bool, (s1, s2), (i, j)::Tuple{Int,Int}) = isinter || s1 < s2 || i < j
isvalidlink(isinter::Bool, (s1, s2), validsublats) =
    isvalidlink(isinter, (s1, s2)) && ((s1, s2) in validsublats || (s2, s1) in validsublats)

function link!(lat::Lattice, lr::LinkRule{AutomaticRangeLinking})
    if nsites(lat) < 200 # Heuristic cutoff
        newlr = convert(LinkRule{SimpleLinking}, lr)
    else
        newlr = convert(LinkRule{TreeLinking}, lr)
    end
    return link!(lat, newlr)
end

function link!(lat::Lattice{T,E,L}, lr::LinkRule{S}) where {T,E,L,S<:LinkingAlgorithm}
    # clearlinks!(lat)
    pre = linkprecompute(lr, lat)
    br = bravaismatrix(lat)
    ndist_zero = zero(SVector{L,Int})
    dist_zero = br * ndist_zero
    lat.links.intralink = buildIlink(lat, lr, pre, (dist_zero, ndist_zero))
    L==0 && return lat

    iter = BoxIterator(Tuple(ndist_zero), maxiterations = lr.maxsteps)
    for cell in iter
        ndist = SVector(cell)

        ndist == ndist_zero && (acceptcell!(iter, cell); continue) # intracell already done
        iswithinmin(cell, lr.mincells) && acceptcell!(iter, cell) # enforce a minimum search range
        isnotlinked(ndist, br, lr) && continue # skip if we can be sure it's not linked

        dist = br * ndist
        ilink = buildIlink(lat, lr, pre, (dist, ndist))
        if !isempty(ilink)
            push!(lat.links.interlinks, ilink)
            acceptcell!(iter, cell)
        end
    end
    return lat
end

@inline iswithinmin(cell, min) = all(abs(c) <= min for c in cell)

# Logic to exlude cells that are not linked to zero cell by any ilink
isnotlinked(ndist, br, lr) = false # default fallback, used unless lr.alg isa BoxIteratorLinking
function isnotlinked(ndist, br, lr::LinkRule{B}) where {T,E,L,NL,B<:BoxIteratorLinking{T,E,L,NL}}
    nm = lr.alg.open2old
    ndist0 = nm * ndist
    linked = all((
        brnorm2 = dot(nm[:,j], nm[:,j]);
        any(abs(dot(ndist0 + ilink.ndist, nm[:,j])) < brnorm2 for ilink in lr.alg.links.interlinks))
        for j in 1:NL)
    return !linked
end

function buildIlink(lat::Lattice{T,E}, lr, pre, (dist, ndist)) where {T,E}
    isinter = any(n -> n != 0, ndist)
    nsl = nsublats(lat)

    slinks = dummyslinks(lat.sublats) # placeholder to be replaced below

    validsublats = matchingsublats(lat, lr)
    for s1 in 1:nsl, s2 in 1:nsl
        isvalidlink(isinter, (s1, s2), validsublats) || continue
        slinks[s2, s1] = buildSlink(lat, lr, pre, (dist, ndist, isinter), (s1, s2))
    end
    return Ilink(ndist, slinks)
end

function buildSlink(lat::Lattice{T,E}, lr, pre, (dist, ndist, isinter), (s1, s2)) where {T,E}
    slinkbuilder = SparseMatrixBuilder(lat, s1, s2)
    for (i, r1) in enumerate(lat.sublats[s1].sites)
        add_neighbors!(slinkbuilder, lr, pre, (dist, ndist, isinter), (s1, s2), (i, r1))
        finalisecolumn!(slinkbuilder)
    end
    return Slink(sparse(slinkbuilder))
end

linkprecompute(linkrules::LinkRule{<:SimpleLinking}, lat::Lattice) =
    lat.sublats

linkprecompute(linkrules::LinkRule{TreeLinking}, lat::Lattice) =
    ([KDTree(sl.sites, leafsize = linkrules.alg.leafsize) for sl in lat.sublats],
     lat.sublats)

linkprecompute(linkrules::LinkRule{<:WrapLinking}, lat::Lattice) =
    nothing

function linkprecompute(lr::LinkRule{<:BoxIteratorLinking}, lat::Lattice)
    # Build an OffsetArray for each sublat s : maps[s] = oa[cells..., iold] = inew, where cells are oldsystem cells, not fill cells
    nslist = lr.alg.nslist
    iterated2old = lr.alg.iterated2old
    maps = [(range = _maprange(boundingboxiter(lr.alg.iter), nsites, iterated2old);
             OffsetArray(zeros(Int, map(length, range)), range))
            for nsites in nslist]
    for (s, register) in enumerate(lr.alg.iter.registers), (inew, (cell, iold)) in enumerate(register.cellinds)
        maps[s][Tuple(iterated2old * SVector(cell))..., iold] = inew
    end
    return maps
end

# Given a iterated2old that is a rectangular identity, this inserts a 0:0 range in the corresponding zero-rows, i.e.
# translates the bounding box to live in the oldsystem cell space instead of the fill cell space
_maprange(bbox::NTuple{2,MVector{N,Int}}, nsites, iterated2old::SMatrix{L,N}) where {L,N} = ntuple(Val(L+1)) do n
    if n <= L
        m = findnonzeroinrow(iterated2old, n)
        if m == 0
            0:0
        else
            bbox[1][m]:bbox[2][m]
        end
    else
        1:nsites
    end
end

function findnonzeroinrow(ss, n)
    for m in 1:size(ss, 2)
      ss[n, m] != 0 && return m
    end
    return 0
end

function add_neighbors!(slinkbuilder, lr::LinkRule{<:SimpleLinking}, sublats, (dist, ndist, isinter), (s1, s2), (i, r1))
    for (j, r2) in enumerate(sublats[s2].sites)
        r2 += dist
        if lr.alg.isinrange(r2 - r1) && isvalidlink(isinter, (s1, s2), (i, j))
            pushtocolumn!(slinkbuilder, j, _rdr(r1, r2))
        end
    end
    return nothing
end

function add_neighbors!(slinkbuilder, lr::LinkRule{TreeLinking}, (trees, sublats), (dist, ndist, isinter), (s1, s2), (i, r1))
    range = lr.alg.range + extended_eps()
    neighs = inrange(trees[s2], r1 - dist, range)
    sites2 = sublats[s2].sites
    for j in neighs
        if isvalidlink(isinter, (s1, s2), (i, j))
            r2 = sites2[j] + dist
            pushtocolumn!(slinkbuilder, j, _rdr(r1, r2))
        end
    end
    return nothing
end

function add_neighbors!(slinkbuilder, lr::LinkRule{<:WrapLinking}, ::Nothing, (dist, ndist, isinter), (s1, s2), (i, r1))
    oldbravais = bravaismatrix(lr.alg.bravais)
    unwrappedaxes = lr.alg.unwrappedaxes
    add_neighbors_wrap!(slinkbuilder, ndist, isinter, i, (s1, s2), lr.alg.links.intralink, oldbravais, unwrappedaxes, true)
    for ilink in lr.alg.links.interlinks
        add_neighbors_wrap!(slinkbuilder, ndist, isinter, i, (s1, s2), ilink, oldbravais, unwrappedaxes, false)
        # This skipdupcheck == false required to exclude interlinks = intralinks in small wrapped lattices
    end
    return nothing
end

function add_neighbors_wrap!(slinkbuilder, ndist, isinter, i, (s1, s2), ilink, oldbravais, unwrappedaxes, skipdupcheck)
    oldslink = ilink.slinks[s2, s1]
    if !isempty(oldslink) && keepelements(ilink.ndist, unwrappedaxes) == ndist
        olddist = oldbravais * zeroout(ilink.ndist, unwrappedaxes)
        for (j, rdr_old) in neighbors_rdr(oldslink, i)
            if isvalidlink(isinter, (s1, s2), (i, j))
                pushtocolumn!(slinkbuilder, j, (rdr_old[1] - olddist / 2, rdr_old[2] - olddist), skipdupcheck)
            end
        end
    end
    return nothing
end

# Notation:
#   celliter = ndist of the filling BoxIterator for a given site i0
#   i0 = index of that site in original lattice (in sublat s1)
#   ndist = ndist of the new unit under consideration (different from the equivalent ndistold)
#   ndold_intercell = that same ndist translated to an ndist in the original lattice, i.e. ndistold
#   ndold_intracell = ndistold of old unitcell containing site i
#   ndold_intracell_shifted = same as ndold_intracell but shifted by -ndist of the new neighboring cell
#   dist = distold of old unit cell containing new site i in the new unit cell
function add_neighbors!(slinkbuilder, lr::LinkRule{<:BoxIteratorLinking}, maps, (dist, ndist, isinter), (s1, s2), (i, r1))
    (celliter, iold) = lr.alg.iter.registers[s1].cellinds[i]
    ndold_intercell = lr.alg.open2old * ndist
    ndold_intracell = lr.alg.iterated2old * SVector(celliter)
    ndold_intracell_shifted = ndold_intracell - ndold_intercell
    dist = bravaismatrix(lr.alg.bravais) * ndold_intracell

    oldlinks = lr.alg.links

    isvalidlink(false, (s1, s2)) &&
        _add_neighbors_ilink!(slinkbuilder, oldlinks.intralink, maps[s2], isinter, (s1, s2), (i, iold, ndold_intracell_shifted), dist)
    for ilink in oldlinks.interlinks
        isvalidlink(true, (s1, s2)) &&
            _add_neighbors_ilink!(slinkbuilder, ilink, maps[s2], isinter, (s1, s2), (i, iold, ndold_intracell_shifted + ilink.ndist), dist)
    end
    return nothing
end

function _add_neighbors_ilink!(slinkbuilder, ilink_old, maps2, isinter, (s1, s2), (i, iold, ndist_old), dist)
    slink_old = ilink_old.slinks[s2, s1]
    isempty(slink_old) && return nothing

    for (jold, rdr_old) in neighbors_rdr(slink_old, iold)
        isvalid = checkbounds(Bool, maps2, Tuple(ndist_old)..., jold)
        if isvalid
            j = maps2[Tuple(ndist_old)..., jold]
            if j != 0 && isvalidlink(isinter, (s1, s2), (i, j))
                pushtocolumn!(slinkbuilder, j, (rdr_old[1] + dist, rdr_old[2]))
            end
        end
    end
    return nothing
end
