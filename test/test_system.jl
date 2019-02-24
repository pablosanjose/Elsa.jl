module SystemTest

using Test
using Elsa, SparseArrays, LinearAlgebra
using Elsa: nlinks, nsites

@test System(:honeycomb, dim = Val(3), htype = Float64, ptype = Float32) isa System{Float64,Float32,3,2}
@test nsites(System(:honeycomb) |> grow(region = Region(:square, 300))) == 207946
@test nsites(System(:square) |> grow(supercell = 31)) == 961
@test nsites(System(:bcc) |> grow(region = Region("spheroid", (10,4,4)))) == 1365

@test nlinks(System(:honeycomb, Model(hopping(1, sublats = (1,2)))) |> grow(region = Region(:square, 30))) == 6094
@test nlinks(System(:honeycomb, Model(hopping(1, ndists = (0,1)))) |> grow(region = Region(:square, 30))) == 3960
@test nlinks(System(:honeycomb, Model(hopping(1, ndists = ((0,1),)))) |> grow(region = Region(:square, 30))) == 3960
@test nlinks(System(:honeycomb, Model(hopping(1, sublats = (1,2), ndists = ((0,1), (0,1))))) |> 
    grow(region = Region(:square, 30))) == 2040
@test nlinks(System(:square, Model(hopping(1, range = 2))) |> grow(supercell = 31)) == 11532
@test nlinks(System(:bcc, Model(hopping(1, range = 1))) |> 
    grow(supercell = ((1, 2, 0),), region = Region("sphere", 10))) == 10868
@test nlinks(System(:bcc, Model(hopping(1, range = 1))) |> 
    grow(supercell = (10, 20, 30), region = Region("sphere", 4))) == 326
@test nlinks(System(:bcc, Model(hopping(1, range = 1))) |> grow(supercell = (10, 20, 30))) == 84000

@test nsites(System("graphene_bilayer", twistindex = 31)) == 11908

@test begin
    sys1 = System(:honeycomb, Model(hopping(1, sublats = (1,2)))) |> grow(region = Region("circle", 7))
    sys2 = System(:square, Model(hopping(1, range = 2))) |> grow(region = Region("circle", 6))
    sys3 = combine(sys1, sys2)
    (nlinks(sys3) == nlinks(sys1) + nlinks(sys2) == 2138) &&
    (nsites(sys3) == nsites(sys1) + nsites(sys2) == 465)
end

@test begin
    sys1 = System(:honeycomb, Model(hopping(1, sublats = (1,2)))) |> grow(region = Region("circle", 7))
    sys2 = System(:square, Model(hopping(1, range = 2))) |> grow(region = Region("circle", 6))
    sys3 = combine(sys1, sys2, Model(hopping(1, range = 1)))
    nlinks(sys3) == 4646
end

@test begin
    sys1 = System(:honeycomb)
    sys2 = transform(sys1, r -> 2r)
    sys2.sublats[1].sites[1] ≈ 2 * sys1.sublats[1].sites[1]
end

# @test Elsa.nlinks(wrap(Lattice(:square, LinkRule(√2), Supercell(2)), exceptaxes = (1,))) == 14
# @test Elsa.nlinks(wrap(Lattice(:square, LinkRule(√2), Supercell(2)), exceptaxes = (2,))) == 14
# @test Elsa.nlinks(wrap(Lattice(:square, LinkRule(√2), Supercell(2)))) == 6

# @test begin
#     lat = mergesublats(Lattice(Preset(:honeycomb_bilayer, twistindex = 2)), (2,1,1,1))
#     Elsa.nlinks(lat) == 32 && Elsa.nsublats(lat) == 2
# end

# @test begin
#     sys = System(Lattice(:square, LinkRule(2), Supercell(3)), Model(Onsite(1), Hopping(.3)))
#     nnz(sys.hbloch.matrix) == 81 && size(sys.vbloch.matrix) == (9, 9)
# end

# @test begin
#     sys = System(Lattice(:honeycomb, LinkRule(1), Supercell(3)), Model(Onsite(1), Hopping(.3, (1,2))))
#     vel = velocity!(sys, k = (.2,.3))
#     ishermitian(vel) && size(vel) == (18, 18)
# end

# @test begin
#     sys = System(Lattice(:honeycomb, LinkRule(1), Region(:square, 5)), Model(Onsite(1), Hopping(.3, (1,2))))
#     iszero(velocity!(sys))
# end

end # module
