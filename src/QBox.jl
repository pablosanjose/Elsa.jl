module QBox

function _transform!() end

using StaticArrays, NearestNeighbors, SparseArrays, LinearAlgebra, OffsetArrays
using Requires

import Base: convert, iterate, IteratorSize, IteratorEltype, eltype, ==
import SparseArrays: sparse!

export Preset, Lattice, Sublat, Bravais, Supercell, LatticeConstant,
       FillRegion, LinkRules, Dim, Precision,
       transform, _transform!, lattice!, combine
export TreeSearch, SimpleSearch
export System, Model, Onsite, Hopping, hamiltonian, MeshBrillouin
export plot
export @SMatrix, @SVector, SMatrix, SVector

# function __init__()
#     # @require Makie="ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" @eval import Makie
#     @require Makie="ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" include("plot.jl")
# end

abstract type LatticeOption end
abstract type LatticePresets end
abstract type ModelTerm end

include("tools.jl")
include("boxiterator.jl")
include("presets.jl")
include("lattice.jl")
include("model.jl")
include("hamiltonian.jl")
include("system.jl")
include("mesh.jl")
include("algorithms_lattice.jl")
include("convert.jl")

# include("plot.jl")
@require Makie="ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" include("plot.jl")

end
