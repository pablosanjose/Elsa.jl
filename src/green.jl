#######################################################################
# GreenFunction
#######################################################################
struct Boundaries{L}
    offsets::NTuple{L, Union{Missing, Int}}
end

struct Perturbation{T,E,L,EL}
    lattice::Lattice{T,E,L,EL}
    cells::Vector{NTuple{L,Int}}
    selfenergy::Matrix{BlochOperator{T,0}}
end

struct GreenFunction{T,N,L,NL}
    bandstructure::Bandstructure{T,N,L,NL}
    boundaries::Boundaries{L}
    perturbations::Vector{Perturbation{T,L}}
end

GreenFunction(sys::System, region::Region, model::Model = sys.model; kw...) =
    GreenFunction(Bandstructure(sys; kw...), region, model)
function GreenFunction(bs::Bandstructure{T,N,L}, region::Region, model::Model; kw...) where {T,N,L}
    
end

function cellsinregion(lat::Lattice{T,E,L}, region::Region{F}) where {T,E,L,F}
    iter = BoxIterator(region.seed)
    cells = NTuple{L,Int}[]
    for cell in iter
        cellpos =
        for sl in 1:numsublats, siten in 1:nsitesub[sl]
            sitepos = cellpos + pos_sites[sl][siten]
            if region.isinregion(sitepos)
                push!(cells, )
