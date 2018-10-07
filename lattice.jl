#######################################################################
# Sublattice (Sublat) : a group of identical sites (e.g. same orbitals)
#######################################################################
struct Sublat{T,E} <: LatticeOption
    name::Union{String,Missing}
    sites::Vector{SVector{E,T}}
end

Sublat(vs...) = Sublat(missing, toSVectors(vs...))
Sublat(name::String, vs::(<:Union{Tuple, AbstractVector{<:Number}})...) = Sublat(name, toSVectors(vs...))
Sublat{T}(vs...) where {T} = Sublat(missing, toSVectors(T, vs...))
Sublat{T}(name::String, vs...) where {T} = Sublat(name, toSVectors(T, vs...))
Sublat{T,E}() where {T,E} = Sublat(missing, SVector{E,T}[])

nsites(s::Sublat) = length(s.sites)
# dim(s::Sublat{T,E}) where {T,E} = E

_transform!(s::S, f::F) where {S<:Sublat, F<:Function} = (s.sites .= f.(s.sites); s)
flatten(ss::Sublat...) = Sublat(ss[1].name, vcat((s.sites for s in ss)...))

#######################################################################
# Bravais
#######################################################################
struct Bravais{T,E,L,EL} <: LatticeOption
    matrix::SMatrix{E,L,T,EL}
end

Bravais(vs...) = Bravais(toSMatrix(vs...))
Bravais{T}(vs...) where {T} = Bravais(toSMatrix(T, vs...))

@inline transform(b::Bravais{T,2,0,0}, f::F) where {T,L,F<:Function} = b
function transform(b::Bravais{T,E,L,EL}, f::F) where {T,E,L,EL,F<:Function}
    vecs = let z = zero(SVector{E,T})
        ntuple(i -> f(b.matrix[:, i]) - f(z), Val(L))
    end
    matrix = hcat(vecs...)
    return Bravais{T,E,L,EL}(matrix)
end

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

Base.isempty(slink::Slink) = isempty(slink.targets)
nlinks(slink::Slink) = length(slink.targets)

# @inline nsites(s::Slink) = isempty(slink) ? error("Unable to extract number of sites from Slink") : length(s.srcpointers) - 1
neighbors_rdr(s::Slink, i) = ((s.targets[j], s.rdr[j]) for j in (s.srcpointers[i]):(s.srcpointers[i+1]-1))
neighbors_rdr(s::Slink) = ((s.targets[j], s.rdr[j]) for i in 1:(length(s.srcpointers)-1) for j in (s.srcpointers[i]):(s.srcpointers[i+1]-1))
sources(s::Slink) = 1:(length(s.srcpointers)-1)

@inline _rdr(r1, r2) = (0.5 * (r1 + r2), r2 - r1)

function _transform!(s::Slink, f::F) where F<:Function 
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
    emptyslink = Slink{T,E}()
    slinks = [ifelse(isvalidlink(isinter, (s1, s2)), Slink{T,E}(nsites(sublats[s1])), emptyslink) 
              for s2 in 1:ns, s1 in 1:ns]
    return Ilink(ndist, slinks)
end

nlinks(ilinks::Vector{<:Ilink}) = isempty(ilinks) ? 0 : sum(nlinks(ilink) for ilink in ilinks)
nlinks(ilink::Ilink) = isempty(ilink.slinks) ? 0 : sum(nlinks(ilink.slinks, i) for i in eachindex(ilink.slinks))
nlinks(ss::Array{<:Slink}, i) = nlinks(ss[i])

Base.isempty(ilink::Ilink) = nlinks(ilink) == 0

_transform!(i::IL, f::F) where {IL<:Ilink, F<:Function} = (_transform!.(i.slinks, f); i)

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

nuniquelinks(links::Links) = nlinks(links.intralink) + nlinks(links.interlinks)
   
nsublats(links::Links) = size(links.intralink.slinks, 1)
# @inline nsiteslist(links::Links) = [nsites(links.intralink.slinks[s, s]) for s in 1:nsublats(links)]

_transform!(l::L, f::F) where {L<:Links, F<:Function} = (_transform!(l.intralink, f); _transform!.(l.interlinks, f); return l)

################################################################################
## Dim LatticeOption
################################################################################
struct Dim{E} <: LatticeOption
end

Dim(e::Int) = Dim{e}()

################################################################################
## Dim LatticeConstant
################################################################################
struct LatticeConstant{T} <: LatticeOption
    a::T
end

################################################################################
## FillRegion LatticeOption
################################################################################
struct FillRegion{E,F<:Function,N} <: LatticeOption
    region::F
    seed::SVector{E, Float64}
    excludeaxes::NTuple{N,Int}
    maxsteps::Int
end

FillRegion(name::Symbol, args...; kw...) = region_presets[name](args...; kw...)

FillRegion{E}(region::F;
    seed::Union{AbstractVector,Tuple} = zeros(SVector{E,Float64}),
    excludeaxes::NTuple{N,Int} = (), maxsteps = 100_000_000) where {E,F,N} =
        FillRegion{E,F,N}(region, SVector(seed), excludeaxes, maxsteps)

################################################################################
##   Precision LatticeOption
################################################################################
struct Precision{T<:Number} <: LatticeOption
    t::Type{T}
end

################################################################################
##   Supercell LatticeOption
################################################################################
"""
    Supercell(inds::NTuple{N,Int}...)
    Supercell(inds::NTuple{N,Int}...)
    Supercell(rescaling::Int)

Define a supercell in terms of a rescaling of the original unit cell, or a new set of
lattice vectors v_i = v0_j * inds_ij. inds_ij are Integers, so that the new lattice 
vectors v_i are commensurate with the old v_j
"""
struct Supercell{S} <: LatticeOption
    matrix::S
end

Supercell(rescaling::Number) = Supercell(Int(rescaling)*I)
Supercell(vs::(<:Union{Tuple,SVector})...) = Supercell(toSMatrix(Int, vs...))
Supercell(rescalings::Vararg{Number,N}) where {N} = Supercell(SMatrix{N,N,Int}(Diagonal(SVector(rescalings))))

#######################################################################
## LinkRules LatticeOption : directives to create links in a lattice
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

struct LinkRules{S<:SearchAlgorithm} <: LatticeOption
    alg::S
    excludesubs::Vector{Tuple{Int,Int}}
    mincells::Int  # minimum range to search in using BoxIterator
    maxsteps::Int
end

LinkRules(alg::S;
    excludesubs = Tuple{Int,Int}[],
    mincells = 0,
    maxsteps = 100_000_000) where S<:SearchAlgorithm =
    LinkRules{S}(alg, excludesubs, abs(mincells), maxsteps)

LinkRules(r = 10.0; range = r,  kw...) = LinkRules(AutomaticRangeSearch(abs(range)); kw...)

LinkRules(l::Links, i::BoxIterator{N}, open2old, iterated2old, bravais, nslist; kw...) where {N} = 
    LinkRules(BoxIteratorSearch(l, i, open2old, iterated2old, bravais, nslist); kw...)
    
#############################  EXPORTED  ##############################
# Lattice : group of sublattices + Bravais vectors + links
#######################################################################
mutable struct Lattice{T, E, L, EL}
    sublats::Vector{Sublat{T,E}}
    bravais::Bravais{T,E,L,EL}
    links::Links{T,E,L}
end
Lattice(sublats::Vector{Sublat{T,E}}, bravais::Bravais{T,E,L}) where {T,E,L} = Lattice(sublats, bravais, emptylinks(sublats, bravais))

Lattice(name::Symbol) = Lattice(Preset(name))
Lattice(name::Symbol, opts...) = lattice!(Lattice(name), opts...)
Lattice(preset::Preset) = lattice_presets[preset.name](; preset.kwargs...)
Lattice(preset::Preset, opts...) = lattice!(Lattice(preset), opts...)
Lattice(opts::LatticeOption...) = lattice!(seedlattice(Lattice{Float64,0,0,0}, opts...), opts...)

# Vararg here is necessary in v0.7-alpha (instead of ...) to make these type-unstable recursions fast
seedlattice(::Type{S}, opts::Vararg{<:LatticeOption,N}) where {S, N} = _seedlattice(seedtype(S, opts...))
_seedlattice(::Type{Lattice{T,E,L,EL}}) where {T,E,L,EL} = Lattice(Sublat{T,E}[], Bravais(zero(SMatrix{E,L,T,EL})))
seedtype(::Type{T}, t, ts::Vararg{<:Any, N}) where {T,N} = seedtype(seedtype(T, t), ts...)
seedtype(::Type{Lattice{T,E,L,EL}}, ::Sublat{T2,E2}) where {T,E,L,EL,T2,E2} = Lattice{T,E2,L,E2*L}
seedtype(::Type{S}, ::Bravais{T2,E,L,EL}) where {T,S<:Lattice{T},T2,E,L,EL} = Lattice{T,E,L,EL}
seedtype(::Type{Lattice{T,E,L,EL}}, ::Dim{E2}) where {T,E,L,EL,E2} = Lattice{T,E2,L,E2*L}
seedtype(::Type{Lattice{T,E,L,EL}}, ::Precision{T2}) where {T,E,L,EL,T2} = Lattice{T2,E,L,EL}
seedtype(::Type{S}, ::FillRegion, ts...) where {S<:Lattice} = S
seedtype(::Type{L}, opt) where {L<:Lattice} = L

vectorsastuples(lat::Lattice) = vectorsastuples(lat.bravais)
vectorsastuples(br::Bravais{T,E,L}) where {T,E,L} = ntuple(l -> round.((br.matrix[:,l]... ,), digits = 6), Val(L))
nsites(lat::Lattice) = isempty(lat.sublats) ? 0 : sum(nsites(sublat) for sublat in lat.sublats)
nsiteslist(lat::Lattice) = [nsites(sublat) for sublat in lat.sublats]
nsublats(lat::Lattice)::Int = length(lat.sublats)
sublatnames(lat::Lattice) = [slat.name for slat in lat.sublats]
nuniquelinks(lat::Lattice) = nuniquelinks(lat.links)
isunlinked(lat::Lattice) = nuniquelinks(lat.links) == 0
@inline bravaismatrix(lat::Lattice) = bravaismatrix(lat.bravais)
@inline bravaismatrix(br::Bravais) = br.matrix
sitegenerator(lat::Lattice) = (site for sl in lat.sublats for site in sl.sites)
linkgenerator_r1r2(ilink::Ilink) = ((rdr[1] -rdr[2]/2, rdr[1] + rdr[2]/2) for s in ilink.slinks for (_,rdr) in neighbors_rdr(s))
selectbravaisvectors(lat::Lattice{T, E}, bools::AbstractVector{Bool}, ::Val{L}) where {T,E,L} =
   Bravais(SMatrix{E,L,T}(lat.bravais.matrix[:,bools]))

supercellmatrix(s::Supercell{<:UniformScaling}, lat::Lattice{T,E,L}) where {T,E,L} = SMatrix{L,L}(s.matrix.λ .* one(SMatrix{L,L,Int}))
supercellmatrix(s::Supercell{<:SMatrix}, lat::Lattice{T,E,L}) where {T,E,L} = s.matrix

function transformlattice!(l::L, f::F) where {L<:Lattice, F<:Function}
    _transform!.(l.sublats, f)
    isunlinked(l) || _transform!(l.links, f)
    l.bravais = transform(l.bravais, f)
    return l
end
transformlattice(l::Lattice, f::F) where F<:Function = transformlattice!(deepcopy(l), f)

#######################################################################
# Apply LatticeOptions
#######################################################################

lattice!(lat::Lattice) = lat

lattice!(lat::Lattice, o1::LatticeOption, o2, opts...) = lattice!(lattice!(lat, o1), o2, opts...)

function lattice!(lat::Lattice, b::Bravais)
    lat.bravais = b
    return lat
end

function lattice!(lat::Lattice, s::Sublat)
    push!(lat.sublats, s)
    return lat
end

lattice!(lat::Lattice{T,E,L,EL}, d::Dim{E2}) where {T,E,L,EL,E2} = convert(Lattice{T,E2,L,E2*L}, lat)

lattice!(lat::Lattice{T,E,L,EL}, p::Precision{T2}) where {T,E,L,EL,T2} = 
    convert(Lattice{T2,E,L,EL}, lat)

lattice!(lat::Lattice, sc::Supercell) = expand_unitcell(lat, sc)

lattice!(lat::Lattice, lr::LinkRules) = link!(lat, lr)

function lattice!(lat::Lattice{T,E,L}, l::LatticeConstant) where {T,E,L}
    if L == 0 
        @warn("Cannot redefine the LatticeConstant of a non-periodic lattice")
    else
        rescale = let factor = l.a / norm(lat.bravais.matrix[:,1])
            r -> factor * r
        end
        _transform!(lat, rescale)
    end
    return lat 
end

function lattice!(lat::Lattice{T,E}, fr::FillRegion{E}) where {T,E}
    fill_region(lat, fr)
end

#######################################################################
# Lattice display
#######################################################################

Base.show(io::IO, lat::Lattice{T,E,L}) where {T,E,L}=
    print(io, "Lattice{$T,$E,$L} : $(L)D lattice in $(E)D space with $T sites
    Bravais vectors : $(vectorsastuples(lat))
    Number of sites : $(nsites(lat))
    Sublattice names : $((sublatnames(lat)... ,))
    Unique Links : $(nuniquelinks(lat))")
