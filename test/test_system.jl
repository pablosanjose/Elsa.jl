module SystemTest

using Test
using QBox, SparseArrays, LinearAlgebra

@test begin
    sys = System(Lattice(:square, Supercell(3), LinkRule(2)), Model(Onsite(1), Hopping(.3)))
    nnz(sys.hbloch.matrix) == 81 && size(sys.vbloch.matrix) == (9, 9)
end

@test begin
    sys = System(Lattice(:honeycomb, Supercell(3), LinkRule(1)), Model(Onsite(1), Hopping(.3, (1,2))))
    vel = velocity(sys, k = (.2,.3))
    ishermitian(vel) && size(vel) == (18, 18)
end

end # module
