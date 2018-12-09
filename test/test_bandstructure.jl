module SystemTest

using Test
using Elsa, SparseArrays, LinearAlgebra

@test begin
    sys = System(Lattice(:honeycomb, Supercell(3), LinkRule(1/√3)), Model(Hopping(1)))
    Bandstructure(sys, partitions = 10).npoints == 100
end


end # module
