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
       link!(newlattice, LinkRules(lat.links, iter, open2old, iterated2old, lat.bravais, nsiteslist(lat);
                                   mincells = cellrange(lat.links)))
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
        link!(newlattice, LinkRules(lat.links, iter, open2old, iterated2old, 
                                    lat.bravais, nsiteslist(lat); mincells = cellrange(lat.links)))
end

function _box_fill(::Val{N}, lat::Lattice{T,E,L}, isinregion::F, fillaxesbool, seed0, maxsteps, usecellaspos) where {N,T,E,L,F}
    seed = convert(SVector{E,T}, seed0)
    fillvectors = SMatrix{E, N}(bravaismatrix(lat)[:, fillaxesbool])
    numsublats = nsublats(lat)
    nregisters = ifelse(isunlinked(lat), 0, numsublats)
    nsitesub = Int[nsites(sl) for sl in lat.sublats]

    # pos_sites::Vector{Vector{SVector{E,T}}} = [[seed + site for site in slat.sites] for slat in lat.sublats]
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

@inline isvalidlink(isinter::Bool, (s1, s2)) = isinter || s1 <= s2
@inline isvalidlink(isinter::Bool, (s1, s2), (i, j)::Tuple{Int,Int}) = isinter || s1 < s2 || i < j
@inline isvalidlink(isinter::Bool, (s1, s2), lr::LinkRules) = isvalidlink(isinter, (s1, s2)) && !((s1, s2) in lr.excludesubs || (s2, s1) in lr.excludesubs)

function clearlinks!(lat::Lattice{T,E,L,EL}) where {T,E,L,EL}
    lat.links = emptylinks(lat)
    return lat
end

function link!(lat::Lattice, lr::LinkRules{AutomaticRangeSearch})
    if nsites(lat) < 200 # Heuristic cutoff
        newlr = convert(LinkRules{SimpleSearch}, lr)
    else
        newlr = convert(LinkRules{TreeSearch}, lr)
    end
    return link!(lat, newlr)
end

function link!(lat::Lattice{T,E,L}, lr::LinkRules{S}) where {T,E,L,S<:SearchAlgorithm}
    clearlinks!(lat)
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
isnotlinked(ndist, br, lr) = false # default fallback, used unless lr.alg isa BoxIteratorSearch
function isnotlinked(ndist, br, lr::LinkRules{B}) where {T,E,L,NL,B<:BoxIteratorSearch{T,E,L,NL}}
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

    emptyslink = Slink{T,E}()
    slinks = fill(emptyslink, nsl, nsl)
    # slinks = Union{Slink{T,E},Nothing}[nothing for _ in 1:nsl, _ in 1:nsl]
    
    for s1 in 1:nsl, s2 in 1:nsl
        isvalidlink(isinter, (s1, s2), lr) || continue
        slinks[s2, s1] = buildSlink(lat, lr, pre, (dist, ndist, isinter), (s1, s2))
    end
    return Ilink(ndist, slinks)
end

function buildSlink(lat::Lattice{T,E}, lr, pre, (dist, ndist, isinter), (s1, s2)) where {T,E}
    slink = Slink{T,E}(nsites(lat.sublats[s1]))
    counter = 1
    for (i, r1) in enumerate(lat.sublats[s1].sites)
        slink.srcpointers[i] = counter
        add_neighbors!(slink, lr, pre, (dist, ndist, isinter), (s1, s2), (i, r1))
        counter = length(slink.targets) + 1
    end
    slink.srcpointers[end] = counter

    # if isempty(slink)
    #     return nothing
    # else
    #     return slink
    # end
    return slink
end

linkprecompute(linkrules::LinkRules{<:SimpleSearch}, lat::Lattice) = 
    lat.sublats
    
linkprecompute(linkrules::LinkRules{TreeSearch}, lat::Lattice) = 
    ([KDTree(sl.sites, leafsize = linkrules.alg.leafsize) for sl in lat.sublats],
     lat.sublats)

function linkprecompute(lr::LinkRules{<:BoxIteratorSearch}, lat::Lattice)
    # Build an OffsetArray for each sublat s : maps[s] = oa[cells..., iold] = inew, where cells are oldsystem cells, not fill cells
    nslist = lr.alg.nslist
    iterated2old = lr.alg.iterated2old
    maps = [(range = _maprange(boundingbox(lr.alg.iter), nsites, iterated2old);
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

function add_neighbors!(slink, lr::LinkRules{SimpleSearch{F}}, sublats, (dist, ndist, isinter), (s1, s2), (i, r1)) where {F}
    for (j, r2) in enumerate(sublats[s2].sites)
        r2 += dist
        if lr.alg.isinrange(r2 - r1) && isvalidlink(isinter, (s1, s2), (i, j))
            push!(slink.targets, j)
            push!(slink.rdr, _rdr(r1, r2))
        end
    end
    return nothing
end

function add_neighbors!(slink, lr::LinkRules{TreeSearch}, (trees, sublats), (dist, ndist, isinter), (s1, s2), (i, r1))
    range = lr.alg.range + extended_eps()
    neighs = inrange(trees[s2], r1 - dist, range)
    sites2 = sublats[s2].sites
    for j in neighs
        if isvalidlink(isinter, (s1, s2), (i, j))
            r2 = sites2[j] + dist
            push!(slink.targets, j)
            push!(slink.rdr, _rdr(r1, r2))
        end
    end
    return nothing
end

function add_neighbors!(slink, lr::LinkRules{<:BoxIteratorSearch}, maps, (dist, ndist, isinter), (s1, s2), (i, r1))
    ndold_intercell = lr.alg.open2old * ndist
    dist_intercell = bravaismatrix(lr.alg.bravais) * ndold_intercell
    (celliter, iold) = lr.alg.iter.registers[s1].cellinds[i]
    ndold_intracell = lr.alg.iterated2old * SVector(celliter)
    Δnold = ndold_intracell - ndold_intercell
    
    oldlinks = lr.alg.links

    isvalidlink(false, (s1, s2), lr) &&
        _add_neighbors_ilink!(slink, oldlinks.intralink, maps[s2], isinter, (s1, s2), 
                              (i, iold, Δnold + oldlinks.intralink.ndist), dist_intercell)
    for ilink in oldlinks.interlinks
        isvalidlink(true, (s1, s2), lr) &&
            _add_neighbors_ilink!(slink, ilink, maps[s2], isinter, (s1, s2), 
                                  (i, iold, Δnold + ilink.ndist), dist_intercell)
    end
    return nothing
end

function _add_neighbors_ilink!(slink, ilink_old, maps2, isinter, (s1, s2), (i, iold, ndist_old), dist_intercell)    
    slink_old = ilink_old.slinks[s2, s1]
    isempty(slink_old) && return nothing
    
    for (jold, rdr_old) in neighbors_rdr(slink_old, iold)
        isvalid = checkbounds(Bool, maps2, Tuple(ndist_old)..., jold)
        if isvalid
            j = maps2[Tuple(ndist_old)..., jold]
            if j != 0 && isvalidlink(isinter, (s1, s2), (i, j))
                push!(slink.targets, j)
                push!(slink.rdr, (rdr_old[1] + dist_intercell, rdr_old[2]))
            end
        end
    end
    return nothing
end


#######################################################################
# Combine lattices
#######################################################################

function combine(lats::Lattice...) 
    combine!(deepcopy.(lats)...)
end
function combine!(lats::Lattice...)
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
    vs1 = MVector(ntuple(i -> b1.matrix[:,i], Val(L))); sort!(vs1)
    vs2 = MVector(ntuple(i -> b2.matrix[:,i], Val(L))); sort!(vs2)
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