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
    r, dr = _rdr(sites[i], sites[j])
    r0 = field.lat.bravais.matrix * dn
    return field.f(r + r0, dr, h)
end