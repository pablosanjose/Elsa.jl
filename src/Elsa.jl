module Elsa

using StaticArrays, NearestNeighbors, SparseArrays, LinearAlgebra, OffsetArrays,
      Arpack, ProgressMeter

import Base: convert, iterate, ==
import SparseArrays: sparse!

export Preset, Sublat, Bravais, systempresets, grow, combine, modifysublats!, transform, transform!
export System, Model, Region, hopping, onsite, hamiltonian
#export apply!, transform, transform!, lattice!, combine, wrap, combinesublats, hamiltonian, velocity, hamiltonian!, velocity!
export @SMatrix, @SVector, SMatrix, SVector

# const NameType = String
# const nametype = string
const NameType = Symbol
const nametype = Symbol

include("presets.jl")
include("lattice.jl")
include("operators.jl")
include("iterators.jl")
include("model.jl")
include("system.jl")
include("system_methods.jl")
# include("mesh.jl")
# include("bandstructure.jl")
include("convert.jl")
include("tools.jl")

# include("plot.jl")
# @require Makie="ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" include("plot.jl")
# @require AbstractPlotting="537997a7-5e4e-5d89-9595-2241ea00577e" include("plot.jl")

end
