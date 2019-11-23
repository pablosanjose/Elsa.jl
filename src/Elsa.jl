module Elsa

using Requires

function __init__()
    # @require Pardiso = "46dd5b70-b6fb-5a00-ae2d-e8fea33afaf2" include("diagonalizers/pardiso.jl")
    # @require KrylovKit = "0b1a1467-8014-51b9-945f-bf0ae24f4b77" include("diagonalizers/krylovkit.jl")
    @require Arpack = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97" include("diagonalizers/arpack.jl")
    # @require ArnoldiMethod = "ec485272-7323-5ecc-a04f-4719b315124d" include("diagonalizers/arnoldimethod.jl")
    # @require IterativeSolvers = "42fd0dbc-a981-5370-80f2-aaf504508153" include("diagonalizers/iterativesolvers.jl")
end

using StaticArrays, NearestNeighbors, SparseArrays, LinearAlgebra, OffsetArrays, Random,
      ProgressMeter
      #FFTW

export sublat, bravais, lattice, dims, hopping, onsite, hamiltonian, randomstate,
       mul!, supercell, unitcell, semibounded, bloch, bloch!, optimize!, similarmatrix,
       sites, bandstructure, marchingmesh

export LatticePresets, RegionPresets

export @SMatrix, @SVector, SMatrix, SVector
export ishermitian, I

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
include("mesh.jl")
include("diagonalizer.jl")
include("bandstructure.jl")
include("convert.jl")
include("tools.jl")

end
