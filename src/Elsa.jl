module Elsa

using StaticArrays, NearestNeighbors, SparseArrays, LinearAlgebra, OffsetArrays
      #FFTW, ProgressMeter, FillArrays

export sublat, bravais, lattice, hopping, onsite, hamiltonian, randomstate,
       mul!, supercell, unitcell, semibounded, bloch, bloch!, optimize!,
       sites

export LatticePresets, RegionPresets

export @SMatrix, @SVector, SMatrix, SVector

const NameType = Symbol
const nametype = Symbol

const TOOMANYITERS = 10^8

include("iterators.jl")
include("presets.jl")
include("lattice.jl")
include("model.jl")
include("field.jl")
include("hamiltonian.jl")
include("state.jl")
# include("operators.jl")
# include("system.jl")
# include("system_methods.jl")
# include("KPM.jl")
# include("mesh.jl")
# include("bandstructure.jl")
include("convert.jl")
include("tools.jl")

end
