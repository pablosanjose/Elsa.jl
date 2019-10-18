module LatticeTest

using Elsa
using Test
using Elsa: nsites, Sublat, Bravais, Lattice

@testset "bravais" begin
    @test bravais() isa Bravais{0,0,Float64,0}
    @test bravais((1, 2), (3, 3)) isa Bravais{2,2,Int,4}
    @test bravais(@SMatrix[1. 2.; 3 3]) isa Bravais{2,2,Float64,4}
end

@testset "sublat" begin
    sitelist = [(3,3), (3,3.), [3,3.], @SVector[3, 3], @SVector[3, 3f0], @SVector[3f0, 3.]]
    for site2 in sitelist, site1 in sitelist
        @test sublat(site1, site2) isa
            Sublat{2,promote_type(typeof.(site1)..., typeof.(site2)...)}
    end
    @test sublat((3,)) isa Sublat{1,Int}
    @test sublat(()) isa Sublat{0,Float64}
end

@testset "lattice" begin
    s = sublat((1, 2))
    for t in (Float32, Float64), e in 1:4, l = 1:4
        b = bravais(ntuple(_ -> (1,), l)...)
        @test lattice(b, s, type = t, dim = Val(e)) isa Lattice{e,min(l,e),t}
        @test lattice(b, s, type = t, dim = e) isa Lattice{e,min(l,e),t}
    end
end

end # module
