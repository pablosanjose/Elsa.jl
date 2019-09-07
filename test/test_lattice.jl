module LatticeTest

using Elsa
using Test
using Elsa: nsites, nlinks, Sublat

@testset "bravais" begin
    @test bravais() isa Bravais{0,0,Float64,0}
    @test bravais((1, 2), (3, 3)) isa Bravais{2,2,Int,4}
    @test bravais(@SMatrix[1. 2.; 3 3]) isa Bravais{2,2,Float64,4}
end

@testset "sublat" begin
    sitelist = [(3,3), (3,3.), [3,3.], @SVector[3, 3], @SVector[3, 3f0], @SVector[3f0, 3.]]
    for site2 in sitelist, site1 in sitelist
        for orbs in [(:a,), (:a,:b), (:a,:b,:a)]
            @test sublat(site1, site2, orbitals = orbs) isa 
                Sublat{2,promote_type(typeof.(site1)..., typeof.(site2)...), length(orbs)}
        end
    end
    @test sublat((3,)) isa Sublat{1,Int,1}
    @test sublat((), orbitals = (:up,:down)) isa Sublat{0,Float64,2}
end

@testset "lattice" begin
    @test_throws InexactError Lattice(Bravais((1, 2.2)), Sublat((1, 2)))
    @test Lattice(Bravais((1, 2.2)), Sublat((1, 2)), ptype = Float32) isa Lattice{2,1,Float32}

    @test Lattice(Bravais((1, 2)), Sublat((1, 2))) isa Lattice{2,1,Int}
    @test Lattice(Bravais((1, 2)), Sublat((1, 2.)), Sublat((2, 1))) isa Lattice{2,1,Float64}
    @test Lattice(Bravais((1, 2)), Sublat((1, 2.)), Sublat((2, 1)); dim = Val(3)) isa Lattice{3,1,Float64}
    @test Lattice(Bravais((1, 2)), Sublat((1, 2.)), Sublat((2, 1)); dim = Val(1)) isa Lattice{1,1,Float64}
    @test Lattice(Bravais((1, 2)), Sublat((1, 2.)), Sublat((2, 1)); dim = Val(3), ptype = Float32) isa Lattice{3,1,Float32}
    @test Lattice(Sublat((1, 2.))) isa Lattice{2,0,Float64}
end

end # module
