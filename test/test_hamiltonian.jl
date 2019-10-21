module HamiltonianTest

using Elsa
using Test
using Elsa: Hamiltonian

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


end # module