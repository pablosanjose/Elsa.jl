module QBox

using StaticArrays, NearestNeighbors, SparseArrays, LinearAlgebra, OffsetArrays,
      Arpack, ProgressMeter

import Base: convert, iterate, ==
import SparseArrays: sparse!

export LatticeDirective, Preset, Lattice, Sublat, Bravais, Supercell, LatticeConstant,
       Dim, FillRegion, LinkRule, TreeLinking, SimpleLinking, Precision
export System, Model, Onsite, Hopping, BrillouinMesh, Bandstructure
export transform, transform!, lattice!, combine, wrap, mergesublats, hamiltonian!, velocity!
export @SMatrix, @SVector, SMatrix, SVector

abstract type LatticeDirective end
abstract type LatticePresets end
abstract type ModelTerm end

include("tools.jl")
include("presets.jl")
include("links.jl")
include("iterators.jl")
include("lattice.jl")
include("model.jl")
include("blochoperator.jl")
include("system.jl")
include("mesh.jl")
include("bandstructure.jl")
include("convert.jl")

# include("plot.jl")
# @require Makie="ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a" include("plot.jl")
# @require AbstractPlotting="537997a7-5e4e-5d89-9595-2241ea00577e" include("plot.jl")

end
