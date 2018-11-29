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

################################## _box_fill() ################################
# fill_region() : fill a region with a lattice
#
# We use a BoxIterator to efficiently fill the region
################################################################################

function fill_region(lat::Lattice{T,E,L}, fr::FillRegion{E,F,N}) where {T,E,L,F,N} #N is number of excludeaxes
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
    SMatrix{L,N,Int}(one(SMatrix{L,L,Int})[:, axesbool])
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

# Notation: celliter = ndist of the filling BoxIterator for a given site i0
#           i0 = index of that site in original lattice (in sublat s1)
#           ndist = ndist of the new unit under consideration (different from the equivalent ndistold)
#           ndold_intercell = that same ndist translated to an ndist in the original lattice, i.e. ndistold
#           ndold_intracell = ndistold of old unitcell containing site i
#           ndold_intracell_shifted = same as ndold_intracell but shifted by -ndist of the new neighboring cell
#           dist = distold of old unit cell containing new site i in the new unit cell
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

#######################################################################
# siteclusters : find disconnected site groups in a sublattice
#######################################################################

function siteclusters(lat::Lattice, sublat::Int, onlyintra)
    isunlinked(lat) && return [Int[]]

    ns = nsites(lat.sublats[sublat])
    sitebins = fill(0, ns)  # sitebins[site] = bin
    binclusters = Int[]     # binclusters[bin] = cluster number
    pending = Int[]

    bincounter = 0
    clustercounter = 0
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
        for neigh in neighbors(lat.links, src, (sublat, sublat), onlyintra)
            if sitebins[neigh] == 0   # unclassified neighbor
                push!(pending, neigh)
                sitebins[neigh] = bincounter; next!(p)
            else
                clustercounter = min(clustercounter, binclusters[sitebins[neigh]])
                binclusters[bincounter] = clustercounter
            end
        end
    end
    clusters = [Int[] for _ in 1:maximum(binclusters)]
    for i in 1:ns
        push!(clusters[binclusters[sitebins[i]]], i)
    end
    return clusters
end
