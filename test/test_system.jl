module SystemTest

using Test
using Elsa, SparseArrays, LinearAlgebra

@test begin
    sys = System(Lattice(:square, LinkRule(2), Supercell(3)), Model(Onsite(1), Hopping(.3)))
    nnz(sys.hbloch.matrix) == 81 && size(sys.vbloch.matrix) == (9, 9)
end

@test begin
    sys = System(Lattice(:honeycomb, LinkRule(1), Supercell(3)), Model(Onsite(1), Hopping(.3, (1,2))))
    vel = velocity!(sys, k = (.2,.3))
    ishermitian(vel) && size(vel) == (18, 18)
end

@test begin
    sys = System(Lattice(:honeycomb, LinkRule(1), Region(:square, 5)), Model(Onsite(1), Hopping(.3, (1,2))))
    iszero(velocity!(sys))
end

end # module
