#######################################################################
# Field
#######################################################################
struct Field{F,N<:NamedTuple,L<:Lattice}
    f::F
    lat::L
    args::N
end

Field(::Missing, lat) = missing
Field(f::F, lat) where {F<:Function} = Field(f, lat, NamedTuple())

#######################################################################
# applyfield
#######################################################################
@inline applyfield(::Missing, h, i, j, dn) = h
function applyfield(field::Field{F}, h, i, j, dn) where {F<:Function}
    sites = field.lat.unitcell.sites
    r0 = field.lat.bravais.matrix * dn
    r, dr = _rdr(sites[j], sites[i] + r0) # rsource, rtarget
    return field.f(r, dr, h)
end