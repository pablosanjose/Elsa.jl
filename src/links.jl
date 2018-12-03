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
    rdr::SparseMatrixCSC{Tuple{SVector{E,T}, SVector{E,T}}, Int}
end

function Slink{T,E}(ntargets::Int, nsources::Int; coordination::Int = E + 1) where {T,E}
    rdr = spzeros(Tuple{SVector{E,T}, SVector{E,T}}, ntargets, nsources)
    sizehint!(rdr.rowval, coordination * nsources)
    sizehint!(rdr.nzval, coordination * nsources)
    return Slink(rdr)
end

Base.getindex(s::Slink, i, j) = Base.getindex(s.rdr, i, j)
Base.setindex!(s::Slink, r, i, j) = Base.setindex!(s.rdr, r, i, j)
Base.zero(::Type{Slink{T,E}}) where {T,E} = dummyslink(Slink{T,E})
Base.isempty(slink::Slink) = (nlinks(slink) == 0)
nlinks(slink::Slink) = nnz(slink.rdr)

nsources(s::Slink) = size(s.rdr, 2)
ntargets(s::Slink) = size(s.rdr, 1)
sources(s::Slink) = 1:nsources(s)
targets(s::Slink) = 1:ntargets(s)

_rdr(r1, r2) = (0.5 * (r1 + r2), r2 - r1)

function transform!(s::Slink, f::F) where F<:Function
    frdr(rdr) = _rdr(f(rdr[1] - 0.5 * rdr[2]), f(rdr[1] + 0.5 * rdr[2]))
    s.rdr.nzval .= frdr.(s.rdr.nzval)
    return s
end

#######################################################################
# Intercell links (Clink) : links between two different unit cells
#######################################################################
struct Ilink{T,E,L}
    ndist::SVector{L,Int} # n-distance of targets
    slinks::Matrix{Slink{T,E}}
end

nlinks(ilinks::Vector{<:Ilink}) = isempty(ilinks) ? 0 : sum(nlinks, ilinks)
nlinks(ilink::Ilink) = isempty(ilink.slinks) ? 0 : sum(i -> nlinks(ilink.slinks, i), eachindex(ilink.slinks))
nlinks(ss::Array{<:Slink}, i) = nlinks(ss[i])
nsublats(ilink::Ilink) = size(ilink.slinks, 1)

Base.isempty(ilink::Ilink) = nlinks(ilink) == 0
transform!(i::IL, f::F) where {IL<:Ilink, F<:Function} = (transform!.(i.slinks, f); i)

#######################################################################
# Links struct
#######################################################################
mutable struct Links{T,E,L}  # mutable to be able to update it with link!
    intralink::Ilink{T,E,L}
    interlinks::Vector{Ilink{T,E,L}}
end

function Base.push!(links::Links, ilink::Ilink)
    if iszero(ilink.ndist)
        links.intralink = ilink
    else
        ind = findfirst(il -> il.ndist == ilink.ndist, links.interlinks)
        ind === nothing ? push!(links.interlinks, ilink) : links.interlinks[ind] = ilink
    end
    return links
end

nlinks(links::Links) = nlinks(links.intralink) + nlinks(links.interlinks)

nsublats(links::Links) = nsublats(links.intralink)
ninterlinks(links::Links) = length(links.interlinks)
allilinks(links::Links) = (getilink(links, i) for i in 0:ninterlinks(links))
getilink(links::Links, i::Int) = i == 0 ? links.intralink : links.interlinks[i]

transform!(l::L, f::F) where {L<:Links, F<:Function} = (transform!(l.intralink, f); transform!.(l.interlinks, f); return l)
