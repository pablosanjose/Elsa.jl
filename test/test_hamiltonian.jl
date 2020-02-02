module HamiltonianTest

using Elsa
using Test
using Elsa: Hamiltonian
using LinearAlgebra: diag

@testset "basic hamiltonians" begin
    presets = (LatticePresets.linear, LatticePresets.square, LatticePresets.triangular,
               LatticePresets.honeycomb, LatticePresets.cubic, LatticePresets.fcc,
               LatticePresets.bcc)
    ts = (1, 2.0, @SMatrix[1 2; 3 4])
    for preset in presets, lat in (preset(), unitcell(preset()))
        E, L = dims(lat)
        dn0 = ntuple(_ -> 1, Val(L))
        for t in ts
            @test hamiltonian(lat, onsite(t) + hopping(t; range = 1)) isa Hamiltonian
            @test hamiltonian(lat, onsite(t) - hopping(t; dn = dn0)) isa Hamiltonian
            @test hamiltonian(lat, onsite(t) + hopping(t; dn = dn0, forcehermitian = false)) isa Hamiltonian
        end
    end
end

@testset "orbitals and sublats" begin
    orbs = (:a, (:a,), (:a, :b, 3), ((:a, :b), :c), ((:a, :b), (:c,)), (Val(2), Val(1)),
            (:A => (:a, :b), :D => :c), :D => Val(4))
    lat = LatticePresets.honeycomb()
    for orb in orbs
        @test hamiltonian(lat, onsite(1), orbitals = orb) isa Hamiltonian
    end
    @test hamiltonian(lat, onsite(1) + hopping(@SMatrix[1 2], sublats = (:A,:B)),
                      orbitals = :B => Val(2)) isa Hamiltonian
    h1 = hamiltonian(lat, onsite(1) + hopping(@SMatrix[1 2], sublats = (:A,:B)),
                      orbitals = :B => Val(2))
    h2 = hamiltonian(lat, onsite(1) + hopping(@SMatrix[1 2], sublats = ((:A,:B),)),
                      orbitals = :B => Val(2))
    @test bloch(h1, 1, 2) == bloch(h2, 1, 2)
end

@testset "fields" begin
    h = LatticePresets.honeycomb() |> hamiltonian(hopping(1) + onsite(0)) |> unitcell(2, onsite! = (o, r) -> 1)
    @test diag(bloch(h)) == ComplexF64[1, 1, 1, 1, 1, 1, 1, 1]
end

end # module