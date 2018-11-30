#######################################################################
# NeighborIterator
#######################################################################

struct NeighborIterator{L}
    l::L
    src::Int
    s1::Int
    s2::Int
end
Base.IteratorSize(::NeighborIterator) = Base.HasLength()
Base.IteratorEltype(::NeighborIterator) = Base.HasEltype()
Base.eltype(::NeighborIterator) = Int
Base.length(ni::NeighborIterator{<:Ilink}) = numneighbors(ni, 0)
function Base.length(ni::NeighborIterator{<:Links})
    l = numneighbors(ni, 0)
    for nilink in 1:ninterlinks(ni.l)
        l += numneighbors(ni, nilink)
    end
    return l
end
numneighbors(ni::NeighborIterator, nlink) = length(nzrange(slink(ni, nlink).rdr, ni.src))

slink(ni::NeighborIterator{<:Links}, nilink) = iszero(nilink) ? ni.l.intralink.slinks[ni.s2, ni.s1] : ni.l.interlinks[nilink].slinks[ni.s2, ni.s1]
slink(ni::NeighborIterator{<:Ilink}, nilink) = ni.l.slinks[ni.s2, ni.s1]
maxilinkindex(ni::NeighborIterator{<:Links}) = ninterlinks(ni.l)
maxilinkindex(ni::NeighborIterator{<:Ilink}) = 0

function iterate(ni::NeighborIterator{<:Links}, state = (0, 1))
    (nilink, ptr) = state
    nilink > maxilinkindex(ni) && return nothing
    s = slink(ni, nilink)
    range = nzrange(s.rdr, ni.src)
    ptr > length(range) && return iterate(ni, (nilink + 1, 1))
    targets = rowvals(s.rdr)
    return (targets[range[ptr]], (nilink, ptr + 1))
end
function iterate(ni::NeighborIterator{<:Ilink}, state = (0, 1, slink(ni, 0)))
    (nilink, ptr, s) = state
    range = nzrange(s.rdr, ni.src)
    targets = rowvals(s.rdr)
    ptr > length(range) ? nothing : @inbounds (targets[range[ptr]], (nilink, ptr + 1, s))
end

NeighborIterator(l::Links, src, sublats, onlyintra::Val{true}) =  NeighborIterator(l.intralink, src, sublats)
NeighborIterator(l::Links, src, sublats, onlyintra::Val{false}) =  NeighborIterator(l, src, sublats)
NeighborIterator(l, src, (s1,s2)::Tuple{Int,Int}) = NeighborIterator(l, src, s1, s2)

neighbors!(ni::NeighborIterator, src) = (ni.src = src; return ni)
neighbors(p...) = NeighborIterator(p...)

neighbors_rdr(s::Slink, src) = ((rowvals(s.rdr)[j], nonzeros(s.rdr)[j]) for j in nzrange(s.rdr, src))
neighbors_rdr(s::Slink) = zip(s.rdr.rowval, s.rdr.nzval)
