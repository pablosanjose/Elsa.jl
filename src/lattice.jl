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
Sublat{T,E}() where {T,E} = Sublat(missing, SVector{E,T}[])

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

@inline transform(b::Bravais{T,E,0,0}, f::F) where {T,E,F<:Function} = b
function transform(b::Bravais{T,E,L,EL}, f::F) where {T,E,L,EL,F<:Function}
    svecs = let z = zero(SVector{E,T})
        ntuple(i -> f(b.matrix[:, i]) - f(z), Val(L))
    end
    matrix = hcat(svecs...)
    return Bravais(matrix)
end

arelinearindep(matrix::SMatrix{E,L}) where {E,L} = E >= L && (L == 0 || fastrank(matrix) == L)

keepaxes(br::Bravais, unwrappedaxes) = Bravais(keepcolumns(bravaismatrix(br), unwrappedaxes))

#######################################################################
# Sublattice links (Slink) : links between two given sublattices
#######################################################################

# Slink follows the same structure as sparse matrices. Field targets are 
# indices of target sites for all origin sites, one after the other. Field
# srcpointers, of length (nsites in s1) + 1, stores the position (offset[n1]) in targets
# where the targets for site n1 start. The last extra element allows to compute
# the target ranges easily for all n1. Finally, rdr is of the same length as targets
# and stores the relative vector positions of n1 and its target.
struct Slink{T,E}
    targets::Vector{Int}
    srcpointers::Vector{Int}
    rdr::Vector{Tuple{SVector{E,T}, SVector{E,T}}} # Better cache locality that r, dr
    function Slink{T,E}(targets, srcpointers, rdr) where {T,E} 
        length(targets) == length(rdr) || throw(DimensionMismatch("Size mismatch in Slink"))
        new(targets, srcpointers, rdr)
    end
end

Slink(t, o, rdr::Vector{Tuple{SVector{E,T}, SVector{E,T}}}) where {T,E} = Slink{T,E}(t, o, rdr)

function Slink{T,E}(nsrcsites = 0; coordination::Int = 2*E) where {T,E}
    targets = Int[]
    rdr = Tuple{SVector{E,T}, SVector{E,T}}[]
    srcpointers = fill(1, nsrcsites + 1)
    sizehint!(targets, coordination * nsrcsites)
    sizehint!(rdr, coordination * nsrcsites)
    return Slink{T,E}(targets, srcpointers, rdr)
end

Base.zero(::Type{Slink{T,E}}) where {T,E} = Slink{T,E}()
Base.isempty(slink::Slink) = isempty(slink.targets)
nlinks(slink::Slink) = length(slink.targets)

function unsafe_pushlink!(slink::Slink, i, j, rdr, skipdupcheck = true)
    if skipdupcheck || !isintail(j, slink.targets, slink.srcpointers[i])
        push!(slink.targets, j)
        push!(slink.rdr, rdr)
    end
    return nothing
end

nsources(s::Slink) = length(s.srcpointers)-1
sources(s::Slink) = 1:nsources(s)
targetrange(s::Slink, src) = isempty(s) ? (1:0) : ((s.srcpointers[src]):(s.srcpointers[src+1]-1))

neighbors(s::Slink, src) = (s.targets[j] for j in targetrange(s, src))
neighbors_rdr(s::Slink, src) = ((s.targets[j], s.rdr[j]) for j in targetrange(s, src))
neighbors_rdr(s::Slink) = ((s.targets[j], s.rdr[j]) for src in sources(s) for j in targetrange(s, src))

@inline _rdr(r1, r2) = (0.5 * (r1 + r2), r2 - r1)

function transform!(s::Slink, f::F) where F<:Function 
    frdr(rdr) = _rdr(f(rdr[1] - 0.5 * rdr[2]), f(rdr[1] + 0.5 * rdr[2]))
    s.rdr .= frdr.(s.rdr)
    return s
end

#######################################################################
# Intercell links (Clink) : links between two different unit cells
#######################################################################
struct Ilink{T,E,L}
    ndist::SVector{L,Int} # n-distance of targets
    slinks::Matrix{Slink{T,E}}
end

function emptyilink(ndist::SVector{L,Int}, sublats::Vector{Sublat{T,E}}) where {T,E,L}
    isinter = !iszero(ndist)
    ns = length(sublats)
    emptyslink = zero(Slink{T,E})
    slinks = fill(emptyslink, ns, ns)
    return Ilink(ndist, slinks)
end

nlinks(ilinks::Vector{<:Ilink}) = isempty(ilinks) ? 0 : sum(nlinks(ilink) for ilink in ilinks)
nlinks(ilink::Ilink) = isempty(ilink.slinks) ? 0 : sum(nlinks(ilink.slinks, i) for i in eachindex(ilink.slinks))
nlinks(ss::Array{<:Slink}, i) = nlinks(ss[i])
nsublats(ilink::Ilink) = size(ilink.slinks, 1)

Base.isempty(ilink::Ilink) = nlinks(ilink) == 0

transform!(i::IL, f::F) where {IL<:Ilink, F<:Function} = (transform!.(i.slinks, f); i)

resizeilink(ilink::IL, ns) where {IL<:Ilink} = IL(ilink.ndist, padrightbottom(ilink.slinks, ns, ns))

#######################################################################
# Links struct
#######################################################################
mutable struct Links{T,E,L}  # mutable to be able to update it with link!
    intralink::Ilink{T,E,L}
    interlinks::Vector{Ilink{T,E,L}} 
end

emptylinks(lat) = emptylinks(lat.sublats, lat.bravais)
emptylinks(sublats::Vector{Sublat{T,E}}, bravais::Bravais{T,E,L}) where {T,E,L} =
    Links(emptyilink(zero(SVector{L, Int}), sublats), Ilink{T,E,L}[])

nlinks(links::Links) = nlinks(links.intralink) + nlinks(links.interlinks)
   
nsublats(links::Links) = nsublats(links.intralink)
# @inline nsiteslist(links::Links) = [nsites(links.intralink.slinks[s, s]) for s in 1:nsublats(links)]

transform!(l::L, f::F) where {L<:Links, F<:Function} = (transform!(l.intralink, f); transform!.(l.interlinks, f); return l)

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
## FillRegion LatticeDirective
################################################################################
"""
    FillRegion(regionname::Symbol, args...)
    FillRegion{E}(region::Function; seed = zero(SVector{E,Float64}), excludeaxes = (), maxsteps = 100_000_000)

Create a `FillRegion{E,F,N} <: LatticeDirective` to fill a region in `E`-dimensional 
space defined by `region(r) == true`, where function `region::F` can alternatively be 
defined by a region `regionname` as `QBox.regionpresets[regionname](args...; kw...)`.

Fill search starts at position `seed`, and takes a maximum of `maxsteps` along all lattice
Bravais vectors, excluding those specified by `excludeaxes::NTuple{N,Int}`.

# Examples
```jldoctest
julia> r = FillRegion(:circle, 10); r.region([10,10])
false

julia> r = FillRegion(:square, 20); r.region([10,10])
true

julia> Tuple(keys(QBox.regionpresets))
(:ellipse, :circle, :sphere, :cuboid, :cube, :rectangle, :spheroid, :square)

julia> r = FillRegion{3}(r -> 0<=r[1]<=1 && abs(r[2]) <= sec(r[1]); excludeaxes = (3,)); r.region((0,1,2))
true
```
"""
struct FillRegion{E,F<:Function,N} <: LatticeDirective
    region::F
    seed::SVector{E, Float64}
    excludeaxes::NTuple{N,Int}
    maxsteps::Int
end

FillRegion(name::Symbol, args...; kw...) = regionpresets[name](args...; kw...)

FillRegion{E}(region::F;
    seed::Union{AbstractVector,Tuple} = zero(SVector{E,Float64}),
    excludeaxes::NTuple{N,Int} = (), maxsteps = 100_000_000) where {E,F,N} =
        FillRegion{E,F,N}(region, SVector(seed), excludeaxes, maxsteps)

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
abstract type SearchAlgorithm end

struct SimpleSearch{F<:Function} <: SearchAlgorithm
    isinrange::F
end
SimpleSearch(range::Number) = SimpleSearch(dr -> norm(dr) <= range + extended_eps())

struct TreeSearch <: SearchAlgorithm
    range::Float64
    leafsize::Int
end
TreeSearch(range; leafsize = 10) = TreeSearch(abs(range), leafsize)

struct WrapSearch{T,E,L,EL,N} <: SearchAlgorithm
    links::Links{T,E,L}
    bravais::Bravais{T,E,L,EL}
    unwrappedaxes::NTuple{N,Int}
end

struct AutomaticRangeSearch <: SearchAlgorithm
    range::Float64
end

struct BoxIteratorSearch{T,E,L,N,EL,O<:SMatrix,C<:SMatrix} <: SearchAlgorithm
    links::Links{T,E,L}
    iter::BoxIterator{N}
    open2old::O
    iterated2old::C
    bravais::Bravais{T,E,L,EL}
    nslist::Vector{Int}
end

"""
    LinkRule(algorithm[, sublats...]; mincells = 0, maxsteps = 100_000_000)
    LinkRule(range[, sublats...]; mincells = 0, maxsteps = 100_000_000))

Create a `LinkRule{S,SL} <: LatticeDirective` to compute links between sites in
sublattices indicated by `sublats::SL` using `algorithm::S <: SearchAlgorithm`. 
`TreeSearch(range::Number)` and `SimpleSearch(isinrange::Function)` are available. 
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
LinkRule{QBox.AutomaticRangeSearch,Tuple{Tuple{Int64,Int64}}}

julia> LinkRule(TreeSearch(2.0)) |> typeof
LinkRule{TreeSearch,Missing}

julia> LinkRule(SimpleSearch(dr -> norm(dr, 1) < 2.0)) |> typeof
LinkRule{SimpleSearch{getfield(Main, Symbol("##9#10"))},Missing}
```
"""
struct LinkRule{S<:SearchAlgorithm, SL} <: LatticeDirective
    alg::S
    sublats::SL
    mincells::Int  # minimum range to search in using BoxIterator
    maxsteps::Int
end
LinkRule(range; kw...) = LinkRule(; range = range, kw...)
LinkRule(range, sublats...; kw...) = LinkRule(; range = range, sublats = sublats, kw...)
LinkRule(; range = 10.0, sublats = missing, kw...) = 
    LinkRule(AutomaticRangeSearch(abs(range)), tuplesort(to_tuples_or_missing(sublats)); kw...)
LinkRule(alg::S; kw...) where S<:SearchAlgorithm = LinkRule(alg, missing; kw...)
LinkRule(alg::S, sublats; mincells = 0, maxsteps = 100_000_000) where S<:SearchAlgorithm =
    LinkRule(alg, sublats, abs(mincells), maxsteps)

#######################################################################
# Lattice : group of sublattices + Bravais vectors + links
#######################################################################
"""
    Lattice([preset, ]directives...)

Build a `Lattice{T,E,L,EL}` of `L` dimensions in `E`-dimensional embedding 
spaceof and composed of `T`-typed sites. Optional `preset::Union{Preset, Symbol}`
is one of the `QBox.latticepresets` or a `Preset(preset, args...)`. 
The `directives::LatticeDirective...` apply additional build instructions in order.

# Examples
```jldoctest
julia> Lattice(Sublat(:C, (1,0), (0,1)), Sublat(:D, (0.5,0.5)), Bravais((1,0), (0,2)), Dim(3))
Lattice{Float64,3,2} : 2D lattice in 3D space with Float64 sites
    Bravais vectors : ((1.0, 0.0, 0.0), (0.0, 2.0, 0.0))
    Number of sites : 3
    Sublattice names : (:C, :D)
    Unique Links : 0

julia> Lattice(:honeycomb, LinkRule(1/√3), FillRegion(:circle, 100))
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

julia> Tuple(keys(QBox.latticepresets))
(:bcc, :graphene, :honeycomb, :cubic, :linear, :fcc, :honeycomb_bilayer, :square, :triangular)
```
"""
mutable struct Lattice{T,E,L,EL}
    sublats::Vector{Sublat{T,E}}
    bravais::Bravais{T,E,L,EL}
    links::Links{T,E,L}
end
Lattice(sublats::Vector{Sublat{T,E}}, bravais::Bravais{T,E,L}) where {T,E,L} = Lattice(sublats, bravais, emptylinks(sublats, bravais))
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
seedtype(::Type{S}, ::FillRegion, ts...) where {S<:Lattice} = S
seedtype(::Type{L}, opt) where {L<:Lattice} = L

vectorsastuples(lat::Lattice) = vectorsastuples(lat.bravais.matrix)
vectorsastuples(br::Bravais) =  vectorsastuples(br.matrix)
vectorsastuples(mat::SMatrix{E,L}) where {E,L} = ntuple(l -> round.((mat[:,l]... ,), digits = 6), Val(L))
nsites(lat::Lattice) = isempty(lat.sublats) ? 0 : sum(nsites(sublat) for sublat in lat.sublats)
nsiteslist(lat::Lattice) = [nsites(sublat) for sublat in lat.sublats]
nsublats(lat::Lattice)::Int = length(lat.sublats)
sublatnames(lat::Lattice) = Union{Symbol,Missing}[slat.name for slat in lat.sublats]
nlinks(lat::Lattice) = nlinks(lat.links)
isunlinked(lat::Lattice) = nlinks(lat.links) == 0
linkspersite(lat::Lattice) = (nlinks(lat.links.intralink) + nlinks(lat.links.interlinks)/2)/nsites(lat)
@inline bravaismatrix(lat::Lattice) = bravaismatrix(lat.bravais)
@inline bravaismatrix(br::Bravais) = br.matrix
sitegenerator(lat::Lattice) = (site for sl in lat.sublats for site in sl.sites)
linkgenerator_r1r2(ilink::Ilink) = ((rdr[1] -rdr[2]/2, rdr[1] + rdr[2]/2) for s in ilink.slinks for (_,rdr) in neighbors_rdr(s))
selectbravaisvectors(lat::Lattice{T, E}, bools::AbstractVector{Bool}, ::Val{L}) where {T,E,L} =
   Bravais(SMatrix{E,L,T}(lat.bravais.matrix[:,bools]))

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

# Ensire the size of Slink matrices in lat.links matches the number of sublats. Do it only at the end.
function adjust_slinks_to_sublats!(lat::Lattice)
    ns = nsublats(lat)
    nsublats(lat.links.intralink) == ns || (lat.links.intralink = resizeilink(lat.links.intralink, ns))
    for (n, ilink) in enumerate(lat.links.interlinks)
        nsublats(ilink) == ns || (lat.links.interlink[n] = resizeilink(ilink, ns))
    end
    return lat
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

function _lattice!(lat::Lattice{T,E}, fr::FillRegion{E}) where {T,E}
    fill_region(lat, fr)
end

#######################################################################
# Lattice display
#######################################################################

Base.show(io::IO, lat::Lattice{T,E,L}) where {T,E,L}=
    print(io, "Lattice{$T,$E,$L}  : $(L)D lattice in $(E)D space with $T sites
    Bravais vectors   : $(vectorsastuples(lat))
    Sublattice names  : $((sublatnames(lat)... ,))
    Total sites       : $(nsites(lat))
    Total links       : $(nlinks(lat))
    Unique links/site : $(linkspersite(lat))")

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
    ilink = emptyilink(first(is).ndist, combined_sublats)
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
            return emptyilink(ndist, lat.sublats)
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
                      LinkRule(WrapSearch(lat.links, lat.bravais, exceptaxes)))
    return newlat
end
