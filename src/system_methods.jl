#######################################################################
# Transform system
#######################################################################
function transform!(sys::System, f::Function; sublatinds = collect(eachindex(sys.lattice.sublats)))
    for s in sublatinds
        sind = sublatindex(sys.sysinfo, s)
        transform!(sys.lattice.sublats[sind], f)
    end
    sys.lattice.bravais = transform(sys.lattice.bravais, f)
    return sys
end
transform!(f::Function; kw...) = sys -> transform!(sys, f; kw...)

transform(sys::System, f; kw...) = transform!(deepcopy(sys), f; kw...)
transform(f::Function; kw...) = sys -> transform(sys, f; kw...)

#######################################################################
# System's Hamiltonian and velocity
#######################################################################
function hamiltonian(sys::System{E,L,T}; 
                     k = zero(SVector{E,T}), ϕn = blochphases(k, sys)) where {E,L,T}
    L == 0 || insertblochphases!(sys.hamiltonian, SVector{L,T}(ϕn))
    return sys.hamiltonian.matrix
end
hamiltonian(; kw...) = sys -> hamiltonian(sys; kw...)

velocity(sys::System{E,L}; kw...) where {E,L} = ntuple(i -> velocity(sys, i; kw...), Val(L))

velocity(sys::System{E,L}, axis::Int; kw...) where {E,L} =
    velocity(sys, SMatrix{L,L,Int}(I)[:, axis]; kw...)

function velocity(sys::System{E,L,T}, dϕaxis::Union{Tuple,SVector}; 
                  k = zero(SVector{E,T}), ϕn = blochphases(k, sys)) where {E,L,T}
    L == 0 || insertblochphases!(sys.velocity, SVector{L,T}(ϕn), SVector{L,T}(dϕaxis))
    return sys.velocity.matrix
end

function blochphases(k, sys::System{E,L,T}) where {E,L,T}
	length(k) == E || throw(DimensionMismatch(
            "The dimension of the Bloch vector `k` should match the embedding dimension $E"))
	return transpose(bravaismatrix(sys)) * SVector{E,T}(k) / (2pi)
end

#######################################################################
# Operator builder tools
#######################################################################
struct IJV{Tv}
    I::Vector{Int}
    J::Vector{Int}
    V::Vector{Tv}
end
IJV{Tv}() where {Tv} = IJV(Int[], Int[], Tv[])

Base.resize!(ijv::IJV, n) = (resize!(ijv.I, n); resize!(ijv.J, n); resize!(ijv.V, n))
Base.append!(ijv::IJV, ijv2::IJV) = 
    (append!(ijv.I, ijv2.I); append!(ijv.J, ijv2.J); append!(ijv.V, ijv2.V);)

function Base.push!(ijv::IJV, s::SMatrix{N,M}, (rowoffset, coloffset)) where {N,M}
    for j in 1:M, i in 1:N
        push!(ijv.I, rowoffset + i)
        push!(ijv.J, coloffset + j)
        push!(ijv.V, s[i,j])
    end
    return nothing
end
Base.isempty(ijv::IJV) = isempty(ijv.I)

struct IJVN{Tv,L}
    IJV::IJV{Tv}
    ndist::SVector{L,Int}
end
IJVN(b::Block) = IJVN(IJV(findnz(b.matrix)...), b.ndist)

struct IJVbuilder{Tv,T,E,L,S<:SystemInfo,EL}
    sysinfo::S
    lattice::Lattice{E,L,T,EL}
    IJVNs::Vector{IJVN{Tv,L}}
    kdtrees::Vector{KDTree{SVector{E,T},Euclidean,T}}
end
IJVbuilder(lat::Lattice{E,L,T}, sysinfo::SystemInfo{Tv}, ) where {E,L,T,Tv} = 
    IJVbuilder(sysinfo, lat, IJVN{Tv,L}[],
        Vector{KDTree{SVector{E,T},Euclidean,T}}(undef, length(lat.sublats)))

function Base.push!(builder::IJVbuilder, ijvn::IJVN)
    n = findfirst(bijvn -> bijvn.ndist == ijvn.ndist, builder.IJVNs)
    if n === nothing
        push!(builder.IJVNs, ijvn)
    else
        append!(builder.IJVNs[n].IJV, ijvn.IJV)
    end
    return builder
end

# This violates DRY with the above, but saves some time and allocations (the above is more critical)
function get_or_add_IJVN(builder::IJVbuilder{Tv}, ndist) where {Tv}
    n = findfirst(ijv -> ijv.ndist == ndist, builder.IJVNs)
    if n === nothing
        ijvn = IJVN(IJV{Tv}(), ndist)
        push!(builder.IJVNs, ijvn)
        return ijvn
    else
        return builder.IJVNs[n]
    end
end

Block(ijvn::IJVN, dimh, sysinfo::SystemInfo) =
    Block(ijvn.ndist, sparse(ijvn.IJV.I, ijvn.IJV.J, ijvn.IJV.V, dimh, dimh), sysinfo)

#######################################################################
# Operator builder
#######################################################################
Operator(lat, sysinfo) = Operator(IJVbuilder(lat, sysinfo))

function Operator(builder::IJVbuilder)
    applyterms!(builder, builder.sysinfo.sampledterms...)
    matrix, intra, inters, boundary = assembleblocks(builder)
    ishermitian(matrix) || @warn "Non-hermitian Hamiltonian!"
    return Operator(matrix, intra, inters, boundary)
end

applyterms!(builder, term, terms...) = applyterms!(applyterm!(builder, term), terms...)
applyterms!(builder) = builder

function applyterm!(builder::IJVbuilder{Tv,T,E,L}, (term, sample, sublats)) where {Tv,T,E,L}
    L == 0 || isonsite(term) || checkinfinite(term)
    ndistiter = ndists(term, Val(L))
    sublatsiter = filteredsublats(sublats, builder.sysinfo)
    ijv = IJV{Tv}()
    ijvc = IJV{Tv}()
    for ndist in ndistiter
        foundndist = false
        addedconjugate = false
        for (s1, s2) in sublatsiter
            isvalidlink(term, (s1, s2), ndist) || continue
            dist = builder.lattice.bravais.matrix * ndist
            for (jsource, rsource) in enumerate(builder.lattice.sublats[s2].sites)
                for itarget in _targets(term, rsource, dist, jsource, (s1, s2), builder)
                    isselfhopping(term, (s1, s2), (itarget, jsource), ndist) && continue
                    rtarget = builder.lattice.sublats[s1].sites[itarget] + dist
                    r, dr   = _rdr(rsource, rtarget)
                    smatrix = term(r, dr)
                    iszero(smatrix) && continue
                    rowoffset = builder.sysinfo.offsets[s1] + (itarget - 1) * builder.sysinfo.norbitals[s1]
                    coloffset = builder.sysinfo.offsets[s2] + (jsource - 1) * builder.sysinfo.norbitals[s2]
                    push!(ijv, smatrix, (rowoffset, coloffset))
                    addconjugate = needsconjugate(term, (s1, s2), ndist)
                    addconjugate && push!(ijvc, smatrix', (coloffset, rowoffset))
                    foundndist = true
                    addedconjugate = addedconjugate || addconjugate
                end
            end
        end
        if foundndist
            acceptcell!(ndistiter, ndist)
            if addedconjugate   # this implies ijv is not empty
                push!(builder, IJVN(ijv, ndist))
                push!(builder, IJVN(ijvc, negSVector(ndist)))
                ijv = IJV{Tv}()
                ijvc = IJV{Tv}()
            elseif !isempty(ijv)
                push!(builder, IJVN(ijv, ndist))
                ijv = IJV{Tv}()
            end
        end
    end
    return builder
end

ndists(::Hopping{F,S,Missing}, ::Val{L}) where {F,S,L} = 
    BoxIterator(zero(SVector{L,Int}))
ndists(h::Hopping{F,S,D}, ::Val{L}) where {F,S,D,L} =
    h.ndists
ndists(::Onsite, ::Val{L}) where {L} = (zero(SVector{L,Int}),)

filteredsublats(::Missing, sysinfo) = allorderedpairs(values(sysinfo.namesdict))
filteredsublats(ss::NTuple{N,Tuple{Int,Int}}, sysinfo) where {N} = unique!(collect(ss))

isselfhopping(term, (s1, s2), (itarget, jsource), ndist) = itarget == jsource && s1 == s2 && !isonsite(term) && iszero(ndist)
isvalidlink(term, (s1, s2), ndist) = needsconjugate(term, (s1, s2), ndist) || (iszero(ndist) && s1 == s2)
needsconjugate(::Onsite, args...) = false
needsconjugate(::Hopping{F,Missing,Missing}, (s1, s2), ndist) where {F} = s1 != s2 || ispositive(ndist)
needsconjugate(::Hopping{F,S,Missing}, (s1, s2), ndist) where {F,S} = s1 != s2 || ispositive(ndist)
needsconjugate(::Hopping{F,Missing}, (s1, s2), ndist) where {F} = s1 != s2 || !iszero(ndist)
needsconjugate(::Hopping, (s1, s2), ndist) = s1 != s2 || !iszero(ndist)
function ispositive(ndist)
    result = false
    for i in ndist
        i == 0 || (result = i > 0; break)
    end
    return result
end

@inline checkinfinite(modelterm) = modelterm.range === missing && modelterm.ndist === missing && 
    throw(ErrorException("Tried to implement an infinite-range hopping on an unbounded lattice"))
function checkblockdims(::SMatrix{N,M}, sdata, sublatsiter) where {N,M}
    @boundscheck for (s1, s2) in sublatsiter
        (N == sdata.norbitals[s1]) && (M == sdata.norbitals[s2]) || 
            throw(DimensionMismatch("Hamiltonian term of dimension ($N,$M) does not match orbital ($(sdata.norbitals[s1]), $(sdata.norbitals[s2]))."))
    end
end

function _targets(term::Hopping{F,S,D,R}, r2, dist, j, (s1, s2), builder) where {F,S,D,R<:Real}
    isassigned(builder.kdtrees, s1) || (builder.kdtrees[s1] = KDTree(builder.lattice.sublats[s1].sites))
    return inrange(builder.kdtrees[s1], r2 - dist, term.range)
end
_targets(term::Hopping{F,S,D,R}, r2, dist, j, (s1, s2), builder) where {F,S,D,R<:Missing} = 
    eachindex(builder.sublats[s1].sites)
_targets(term::Onsite, r2, dist, j, (s1, s2), builder) = (j,)

function assembleblocks(builder::IJVbuilder{Tv,T,E,L}) where {Tv,T,E,L}
    dimh = sum(builder.sysinfo.dims)
    intraIJVN = get_or_add_IJVN(builder, zero(SVector{L,Int}))
    intra = Block(intraIJVN, dimh, builder.sysinfo)
    inters = Block{Tv,L}[]
    for ijvn in builder.IJVNs
        iszero(ijvn.ndist) || push!(inters, Block(ijvn, dimh, builder.sysinfo))
    end
    ijv = intraIJVN.IJV
    foreach(ijvn -> iszero(ijvn.ndist) || appendIJV!(ijv, ijvn.IJV), builder.IJVNs)
    matrix = sparse(ijv.I, ijv.J, ijv.V, dimh, dimh)
    # matrix = +(intra.matrix, [inter.matrix for inter in inters]...)  ## This is a bit slower
    boundary = extractboundary(matrix, intra, inters)
    return matrix, intra, inters, boundary
end
appendIJV!(ijv, ijv2::IJV) = (append!(ijv.I, ijv2.I); append!(ijv.J, ijv2.J); append!(ijv.V, ijv2.V))
 
function extractboundary(matrix, intra::Block{Tv}, inters) where {Tv}
    rowsintra = rowvals(intra.matrix)
    rowsmatrix = rowvals(matrix)
    dimh = size(matrix, 1)
    boundary = Tuple{Int,Int}[]
    for col = 1:dimh 
        for inter in inters
            rowsn = rowvals(inter.matrix)
            for ptr in nzrange(inter.matrix, col)
                ptrintra = ptrmatrix = 0
                row = rowsn[ptr]
                for ptrm in nzrange(matrix, col) 
                    rowsmatrix[ptrm] == row && (ptrmatrix = ptrm; break)
                end
                iszero(ptrmatrix) && throw(ErrorException(
                    "Unexpected: found element in intercell harmonic not present in work matrix"))
                for ptri in nzrange(intra.matrix, col) 
                    rowsintra[ptri] == row && (ptrintra = ptri; break)
                end
                push!(boundary, (ptrmatrix, ptrintra))
                # ptrintra == 0 if inter element is only present in ptrmatrix
            end
        end
    end
    return unique!(sort!(boundary))
end

#######################################################################
# combine systems
#######################################################################
combine(ss...; kw...) = _combine(collectfirst(ss...)...; kw...)
combine(model::Model) = sys -> combine(sys, model)

_combine(systems, (m,)::Tuple{Model}; kw...) = _combine(systems, promote_model(m, systems...); kw...)
function _combine(systems::NTuple{N,System}, model::Model; kw...) where {N}
    sublats = _combinesublats(systems...; kw...)
    bravais = first(systems).lattice.bravais
    lattice = _lattice(bravais, sublats)
    sampledterms = _getsampledterms(systems...)
    sysinfo = SystemInfo(lattice, model, sampledterms...)
    sysinfo_withoutnew = SystemInfo(lattice, model)
    _copyoffsets!(sysinfo_withoutnew, sysinfo)
    builder = IJVbuilder(lattice, sysinfo_withoutnew)
    nsublats = 1
    for system in systems
        offset = sysinfo.offsets[nsublats]
        _combine_block(builder, system.hamiltonian.intra, offset)
        foreach(block -> _combine_block(builder, block, offset), system.hamiltonian.inters)
        nsublats += length(system.lattice.sublats)
    end
    hamiltonian = Operator(builder)
    velocity = boundaryoperator(hamiltonian)
    return System(lattice, hamiltonian, velocity, sysinfo)
end

_getsampledterms(system, systems...) = (system.sysinfo.sampledterms..., _getsampledterms(systems...)...)
_getsampledterms() = ()  

function _copyoffsets!(sysinfo1, sysinfo2)
    copy!(sysinfo1.norbitals, sysinfo2.norbitals)
    copy!(sysinfo1.dims, sysinfo2.dims)
    copy!(sysinfo1.offsets, sysinfo2.offsets)
end

function _combinesublats(system::System{Tv,T,E}, systems...; checkbravais = true) where {Tv,T,E}
    !checkbravais || is_bravais_compatible(system, systems...) || throw(ErrorException("Systems with incompatible Bravais matrices"))
    sublats = deepcopy(system.lattice.sublats)
    foreach(sys -> append!(sublats, sys.lattice.sublats), systems)
    return sublats
end

_combine(systems, ::Tuple{}; kw...) = _combine(systems, Model(); kw...)

function _combine_block(builder, block, offset)
    ijvn = IJVN(block)
    ijvn.IJV.I .+= offset
    ijvn.IJV.J .+= offset
    push!(builder, ijvn)
end

is_bravais_compatible() = true
is_bravais_compatible(sys::System, ss::System...) = all(s -> isequal(sys.lattice.bravais, s.lattice.bravais), ss) 
 
function Base.isequal(b1::Bravais{E,L}, b2::Bravais{E,L}) where {E,L}
    vs1 = ntuple(i -> b1.matrix[:, i], Val(L))
    vs2 = ntuple(i -> b2.matrix[:, i], Val(L))
    for v2 in vs2
        found = false
        for v1 in vs1
            (isapprox(v1, v2) || isapprox(v1, -v2)) && (found = true; break)
        end
        !found && return false
    end
    return true
end

#######################################################################
# Grow system
#######################################################################
struct BlockBuilder{Tv,L}
    ndist::SVector{L,Int}
    matrixbuilder::SparseMatrixBuilder{Tv}
end
BlockBuilder{Tv,L}(ndist::SVector, n::Int) where {Tv,L} = BlockBuilder{Tv,L}(ndist, SparseMatrixBuilder{Tv}(n, n))
BlockBuilder{Tv,L}(n::Int) where {Tv,L} = BlockBuilder{Tv,L}(zero(SVector{L,Int}), SparseMatrixBuilder{Tv}(n, n))
Block(bb::BlockBuilder, sysinfo) = Block(bb.ndist, sparse(bb.matrixbuilder), sysinfo)

struct OperatorBuilder{Tv,L}
    intra::BlockBuilder{Tv,L}
    inters::Vector{BlockBuilder{Tv,L}}
end
OperatorBuilder{Tv,L}(n::Int) where {Tv,L} =
    OperatorBuilder(BlockBuilder{Tv,L}(n), BlockBuilder{Tv,L}[])
dim(o::OperatorBuilder) = size(o.intra.matrixbuilder, 1)

function Operator(b::OperatorBuilder, sysinfo) 
    intra = Block(b.intra, sysinfo)
    inters = [Block(inter, sysinfo) for inter in b.inters]
    matrix = +(intra.matrix, [inter.matrix for inter in inters]...)
    boundary = extractboundary(matrix, intra, inters)
    return Operator(matrix, intra, inters, boundary)
end

_truefunc(r) = true
grow(; kw...) = sys -> grow(sys; kw...)
grow(sys::System{E,L}; supercell = SMatrix{L,0,Int}(), region = _truefunc) where {E,L} = 
    grow(sys, supercell, region)
grow(sys::System{E,L}, supercell::Integer, region) where {E,L} = 
    grow(sys, SMatrix{L,L,Int}(supercell*I), region)
grow(sys::System{E,L}, supercell::NTuple{L,Integer}, region) where {E,L} = 
    grow(sys, SMatrix{L,L,Int}(Diagonal(SVector{L,Int}(supercell))), region)
grow(sys::System{E,L}, supercell::NTuple{N,NTuple{L,Integer}}, region) where {E,L,N} = 
    grow(sys, toSMatrix(supercell...), region)

grow(sys::System, supercell, region) = 
    throw(DimensionMismatch("Possible mismatch between `supercell` and system, or `region` not a function."))

function grow(sys::System{E,L}, supercell::SMatrix{L,L2,Int}, region::Function) where {E,L,L2}
    L2 < L && region == _truefunc && error("Unbounded fill region for $(L-L2) dimensions")
    sublats, sitemaps = _growsublats(sys, supercell, region)
    bravais = _growbravais(sys, supercell)
    sysinfo = _growsysinfo(sys.sysinfo, sublats)
    hamiltonian = _growhamiltonian(sys, supercell, sitemaps, sysinfo)
    velocity = boundaryoperator(hamiltonian)
    return System(Lattice(sublats, bravais), hamiltonian, velocity, sysinfo)
end

_growbravais(sys::System{Tv,T,E,L}, supercell) where {Tv,T,E,L} = 
    Bravais(bravaismatrix(sys) * supercell)

function _growsysinfo(oldsysinfo::SystemInfo, sublats)
    sysinfo = deepcopy(oldsysinfo)
    for (i, sublat) in enumerate(sublats)
        sysinfo.nsites[i] = length(sublat.sites)
        sysinfo.dims[i] = sysinfo.norbitals[i] * sysinfo.nsites[i]
        sysinfo.offsets[i + 1] = sysinfo.offsets[i] + sysinfo.dims[i]
    end
    return sysinfo
end

const TOOMANYITERS = 10^8
function _growsublats(sys::System{E,L,T}, supercell, region) where {E,L,T}
    bravais = bravaismatrix(sys)
    iter = BoxIterator(zero(SVector{L,Int}))
    hasnoshadow = _hasnoshadow(supercell)
    firstfound = false
    counter = 0
    for ndist in iter   # We first compute the bounding box
        found = false
        counter += 1; counter == TOOMANYITERS && @warn "Region fill seems non-covergent, check `region`"
        for sublat in sys.lattice.sublats, site in sublat.sites
            r = bravais * ndist + site
            found = hasnoshadow(ndist) && region(r)
            if found || !firstfound
                acceptcell!(iter, ndist)
                firstfound = found
                break
            end
        end
    end
    bbox = boundingboxiter(iter)
    ranges = UnitRange.(bbox...)

    # sitemaps[sublat][oldsiteindex, cell...] is newsiteindex for a given sublat
    sitemaps = OffsetArray{Int,L+1,Array{Int,L+1}}[zeros(Int, 1:length(sublat.sites), 
        ranges...) for sublat in sys.lattice.sublats]
    
    sublats = [Sublat{E,T}(; name = sublat.name) for sublat in sys.lattice.sublats]
    for (s, sublat) in enumerate(sys.lattice.sublats)
        counter = 0
        for cell in CartesianIndices(ranges) , (i, site) in enumerate(sublat.sites)
            ndist = SVector{L,Int}(Tuple(cell))
            r = site + bravais * ndist
            if hasnoshadow(ndist) && region(r)
                push!(sublats[s].sites, r)
                counter += 1
                sitemaps[s][i, Tuple(cell)...] = counter
            end
        end
    end
    return sublats, sitemaps
end

pinvint(s::SMatrix{N,0}) where {N} = (SMatrix{0,0,Int}(), 0)
function pinvint(s::SMatrix{N,M}) where {N,M}
    qrfact = qr(s)
    pinverse = inv(qrfact.R) * qrfact.Q'
    n = det(qrfact.R)^2
    iszero(n) && throw(ErrorException("Supercell is singular"))
    return round.(Int, n * inv(qrfact.R) * qrfact.Q'), round(Int, n)
end
# This is true whenever old ndist is perpendicular to new lattice
_hasnoshadow(supercell) = let invs = pinvint(supercell); ndist -> iszero(_newndist(ndist, invs)); end
_newndist(oldndist, (pinvs, n)) = fld.(pinvs * oldndist, n)
_newndist(oldndist, (pinvs, n)::Tuple{<:SMatrix{0,0},Int}) = SVector{0,Int}()
_wrappedndist(ndist, s, invs) = (nn = _newndist(ndist, invs); (nn, ndist - s * nn))


function _growhamiltonian(sys::System{E,L,T,Tv}, supercell::SMatrix{L,L2}, sitemaps, newsysinfo) where {E,L,T,Tv,L2}
    blocks = sort!(append!([sys.hamiltonian.intra], sys.hamiltonian.inters), by = block -> reverse(block.ndist))
    dimh = sum(s -> sys.sysinfo.norbitals[s] * maximum(sitemaps[s]), eachindex(sys.lattice.sublats))
    invsupercell = pinvint(supercell)
    opbuilder = OperatorBuilder{Tv,L2}(dimh)
    colsdone = 0
    for (s2, sublat2) in enumerate(sys.lattice.sublats)
        cartesian_s2 = CartesianIndices(sitemaps[s2])
        norb2 = sys.sysinfo.norbitals[s2]
        offset2 = sys.sysinfo.offsets[s2]
        for (j, newsitesrc) in enumerate(sitemaps[s2])
            iszero(newsitesrc) && continue  # skip column, source not in region
            ndistsrc = SVector{L,Int}(Base.tail(Tuple(cartesian_s2[j])))
            sitesrc = first(Tuple(cartesian_s2[j]))
            for orb2 in 1:norb2
                col = offset2 + (sitesrc - 1) * norb2 + orb2
                for block in blocks
                    ptrs = nzrange(block.matrix, col)
                    isempty(ptrs) && continue
                    rows = rowvals(block.matrix)
                    vals = nonzeros(block.matrix)
                    ndist = ndistsrc + block.ndist
                    newndist, wrappedndist = _wrappedndist(ndist, supercell, invsupercell)
                    newblock = get_or_add_blockbuilder(opbuilder, newndist, colsdone)
                    for ptr in ptrs
                        site, orb, s1 = tosite(rows[ptr], sys.sysinfo)
                        checkbounds(Bool, sitemaps[s1], site, Tuple(wrappedndist)...) || continue
                        newsitedest = sitemaps[s1][site, Tuple(wrappedndist)...]
                        iszero(newsitedest) && continue
                        row = torow(newsitedest, s1, newsysinfo) + orb - 1
                        pushtocolumn!(newblock.matrixbuilder, row, vals[ptr])
                    end
                end
                colsdone += 1
                finalisecolumn!(opbuilder.intra.matrixbuilder)
                foreach(block -> finalisecolumn!(block.matrixbuilder), opbuilder.inters)
            end
        end
    end
    return Operator(opbuilder, newsysinfo)
end

function get_or_add_blockbuilder(op::OperatorBuilder{Tv,L}, ndist, colsdone) where {Tv,L}
    if iszero(ndist)
        return op.intra
    else
        k = findfirstblock(op.inters, ndist)  # better than findfirst (less runtime variance)
        if iszero(k)    # not found, add a new block builder
            block = BlockBuilder{Tv,L}(ndist, dim(op))
            finalisecolumn!(block.matrixbuilder, colsdone)
            push!(op.inters, block)
            return block
        else
            return op.inters[k]
        end
    end
end
function findfirstblock(blocks, ndist)
    for n in eachindex(blocks)
        blocks[n].ndist == ndist && return n
    end
    return 0
end

#######################################################################
# bound
#######################################################################
bound(;kw...) = sys -> bound(sys; kw...)
bound(sys::System{E,L}; except = ()) where {E,L,L2} = 
    _bound(sys, toSVector(Int, except))
function _bound(sys::System{E,L,T,Tv}, exceptaxes::SVector{L2,Int}) where {E,L,L2,T,Tv} 
    sublats = deepcopy(sys.lattice.sublats)
    bravais = Bravais(sys.lattice.bravais.matrix[:, exceptaxes])
    lattice = Lattice(sublats, bravais)
    sysinfo = sys.sysinfo
    matrix = deepcopy(sys.hamiltonian.matrix)
    supercell = SMatrix{L,L,Int}(I)[:,exceptaxes]
    intra =   _projectblock(sys.hamiltonian.intra, supercell)
    inters = Block{Tv,L2}[_projectblock(inter, supercell) for inter in sys.hamiltonian.inters 
              if _isselfshadow(inter.ndist, exceptaxes)]
    boundary = extractboundary(matrix, intra, inters)
    ham = Operator(matrix, intra, inters, boundary)
    vel = boundaryoperator(ham)
    return System(lattice, ham, vel, sysinfo)         
end

# This is true if the part of ndist orthogonal to exceptaxes is zero
_isselfshadow(ndist::SVector{L}, exceptaxes) where {L} = 
    all(i -> iszero(ndist[i]) || i in exceptaxes, 1:L)

function _projectblock(block, scell) 
    Block(scell' * block.ndist, copy(block.matrix), block.nlinks)
end