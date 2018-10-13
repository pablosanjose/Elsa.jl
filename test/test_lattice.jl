module LatticeTest

using Test
using QBox 
using QBox: nsites, nuniquelinks

@test Bravais() isa Bravais{Float64,0,0,0}
@test Bravais((1,2),(3,3)) isa Bravais{Int64,2,2,4}
@test Bravais([1.,2.],[3,3]) isa Bravais{Float64,2,2,4}
@test Bravais([1.,2.], @SVector [3,3]) isa Bravais{Float64,2,2,4}
@test Bravais(@SMatrix [1. 2.; 3 3]) isa Bravais{Float64,2,2,4}

@test Bravais{Float32}() isa Bravais{Float32,0,0,0}
@test Bravais{Float32}((1,2),(3,3)) isa Bravais{Float32,2,2,4}
@test Bravais{Float32}([1.,2.],[3.,3]) isa Bravais{Float32,2,2,4}
@test Bravais{Float32}([1.,2.], @SVector [3,3]) isa Bravais{Float32,2,2,4}
@test Bravais{Float32}(@SMatrix [1. 2.; 3 3]) isa Bravais{Float32,2,2,4}

@test Bravais(Float32[1.,2.], @SVector [3f0,3f0]) isa Bravais{Float32,2,2,4}

@test Sublat() isa Sublat{Float64,0}
@test Sublat((3,3)) isa Sublat{Int64,2}
@test Sublat((3,3.)) isa Sublat{Float64,2}
@test Sublat([3,3.]) isa Sublat{Float64,2}
@test Sublat(@SVector[3,3]) isa Sublat{Int64,2}
@test Sublat(@SVector[3.,3]) isa Sublat{Float64,2}
@test Sublat(@SVector[3.,3], (3,3)) isa Sublat{Float64,2}
@test Sublat([3,4.], [3,3]) isa Sublat{Float64,2}
@test Sublat((3,4.), [3,3]) isa Sublat{Float64,2}
@test Sublat(@SVector[3f0,3f0]) isa Sublat{Float32,2}

@test Sublat{Float32}() isa Sublat{Float32,0}
@test Sublat{Float32}((3,3)) isa Sublat{Float32,2}
@test Sublat{Float32}((3,3.)) isa Sublat{Float32,2}
@test Sublat{Float32}([3,3.]) isa Sublat{Float32,2}
@test Sublat{Float32}(@SVector[3,3]) isa Sublat{Float32,2}
@test Sublat{Float32}(@SVector[3.,3]) isa Sublat{Float32,2}
@test Sublat{Float32}(@SVector[3.,3], (3,3)) isa Sublat{Float32,2}
@test Sublat{Float32}([3,4.], [3,3]) isa Sublat{Float32,2}
@test Sublat{Float32}((3,4.), [3,3]) isa Sublat{Float32,2}
@test Sublat{Float64}(@SVector[3f0,3f0]) isa Sublat{Float64,2}

@test Sublat("A", @SVector[3f0,3f0], (3,4)) isa Sublat{Float32,2}
@test Sublat{Float64}("A", (3f0,3)) isa Sublat{Float64,2}

@test Lattice(:honeycomb, Dim(3), Precision(Float32)) isa Lattice{Float32,3,2}
@test QBox.nsites(Lattice(:honeycomb, FillRegion(:square, 300))) == 207946
@test QBox.nsites(Lattice(:square, Supercell(31))) == 961
@test QBox.nsites(Lattice(:bcc, FillRegion(:spheroid, (10,4,4)))) == 1365

@test QBox.nuniquelinks(Lattice(:honeycomb, LinkRules(1/√3), FillRegion(:square, 300))) == 311273
@test QBox.nuniquelinks(Lattice(:square, Supercell(31), LinkRules(2))) == 6074
@test QBox.nuniquelinks(Lattice(:square, LinkRules(2), Supercell(31))) == 6074
@test QBox.nuniquelinks(Lattice(:bcc, LinkRules(1), FillRegion(:spheroid, (10,4,4)))) == 8216

@test begin
    lat1 = Lattice(:honeycomb, LinkRules(1/sqrt(3)), FillRegion(:circle, 7))
    lat2 = Lattice(:square, LinkRules(2), FillRegion(:circle, 6))
    lat3 = combine(lat1, lat2)
    (nuniquelinks(lat3) == nuniquelinks(lat1) + nuniquelinks(lat2)) &&
    (nsites(lat3) == nsites(lat1) + nsites(lat2))
end

@test begin
    lat1 = Lattice(:honeycomb, LinkRules(1/sqrt(3)))
    lat2 = combine(lat1, lat1)
    _transform!(lat1, r -> 2r)
    isapprox(lat1.sublats[1].sites[1], 2 * lat2.sublats[1].sites[1])
end

end
