#######################################################################
# Lattice presets
#######################################################################

module LatticePresets

using Elsa
using Elsa: NameType

linear(; a0 = 1, semibounded = false, kw...) =
    lattice(a0 * bravais((1.,); kw...), sublat((0.,)); kw...)

square(; a0 = 1, kw...) =
    lattice(a0 * bravais((1., 0.), (0., 1.); kw...), sublat((0., 0.)); kw...)

triangular(; a0 = 1, kw...) =
    lattice(a0 * bravais(( cos(pi/3), sin(pi/3)),(-cos(pi/3), sin(pi/3)); kw...),
        sublat((0., 0.)); kw...)

honeycomb(; a0 = 1, kw...) =
    lattice(a0 * bravais((cos(pi/3), sin(pi/3)), (-cos(pi/3), sin(pi/3)); kw...),
        sublat((0.0, -0.5/sqrt(3.0)), name = :A),
        sublat((0.0,  0.5/sqrt(3.0)), name = :B); kw...)

cubic(; a0 = 1, kw...) =
    lattice(a0 * bravais((1., 0., 0.), (0., 1., 0.), (0., 0., 1.); kw...),
        sublat((0., 0., 0.)); kw...)

fcc(; a0 = 1, kw...) =
    lattice(a0 * bravais(@SMatrix([-1. -1. 0.; 1. -1. 0.; 0. 1. -1.])'/sqrt(2.); kw...),
        sublat((0., 0., 0.)); kw...)

bcc(; a0 = 1, kw...) =
    lattice(a0 * bravais((1., 0., 0.), (0., 1., 0.), (0.5, 0.5, 0.5); kw...),
        sublat((0., 0., 0.)); kw...)

end # module

#######################################################################
# Hamiltonian presets
#######################################################################

module HamiltonianPresets

using Elsa, LinearAlgebra

function graphene_bilayer(; twistindex = 1, twistindices = (twistindex, 1), a0 = 0.246,
                            interlayerdistance = 1.36a0, rangeintralayer = a0/sqrt(3),
                            rangeinterlayer = 4a0/sqrt(3), hopintra = 2.70, hopinter = 0.48,
                            kwsys...)
    (m, r) = twistindices
    θ = acos((3m^2 + 3m*r +r^2/2)/(3m^2 + 3m*r + r^2))
    sAbot = sublat((0.0, -0.5a0/sqrt(3.0), - interlayerdistance / 2); name = :Ab)
    sBbot = sublat((0.0,  0.5a0/sqrt(3.0), - interlayerdistance / 2); name = :Bb)
    sAtop = sublat((0.0, -0.5a0/sqrt(3.0),   interlayerdistance / 2); name = :At)
    sBtop = sublat((0.0,  0.5a0/sqrt(3.0),   interlayerdistance / 2); name = :Bt)
    br = a0 * bravais(( cos(pi/3), sin(pi/3), 0),
                           (-cos(pi/3), sin(pi/3), 0))
    if gcd(r, 3) == 1
        scbot = @SMatrix[m -(m+r); (m+r) 2m+r]
        sctop = @SMatrix[m+r -m; m 2m+r]
    else
        scbot = @SMatrix[m+r÷3 -r÷3; r÷3 m+2r÷3]
        sctop = @SMatrix[m+2r÷3 r÷3; -r÷3 m+r÷3]
    end
    lattop = lattice(br, sAtop, sBtop; dim = Val(3))
    latbot = lattice(br, sAbot, sBbot; dim = Val(3))
    modelintra = hopping(hopintra, range = rangeintralayer)
    htop = hamiltonian(lattop, modelintra; kwsys...) |> unitcell(sctop)
    hbot = hamiltonian(latbot, modelintra; kwsys...) |> unitcell(scbot)
    let R = @SMatrix[cos(θ/2) -sin(θ/2) 0; sin(θ/2) cos(θ/2) 0; 0 0 1]
        transform!(htop, r -> R * r)
    end
    let R = @SMatrix[cos(θ/2) sin(θ/2) 0; -sin(θ/2) cos(θ/2) 0; 0 0 1]
        transform!(hbot, r -> R * r)
    end
    modelinter = hopping(
        (r,dr) -> hopinter * exp(-3*(norm(dr)/interlayerdistance - 1)) * dr[3]^2/sum(abs2,dr),
        range = rangeinterlayer,
        sublats = ((:Ab,:At), (:Ab,:Bt), (:Bb,:At), (:Bb,:Bt)))
    return combine(hbot, htop, modelinter)
end

end # module

#######################################################################
# Region presets
#######################################################################

module RegionPresets

using StaticArrays

struct Region{E,F}
    f::F
end

Region{E}(f::F) where {E,F<:Function} = Region{E,F}(f)

(region::Region{E})(r::SVector{E2}) where {E,E2} = region.f(r)

Base.show(io::IO, ::Region{E}) where {E} =
    print(io, "Region{$E} : region in $(E)D space")

extended_eps(T = Float64) = sqrt(eps(T))

circle(radius = 10.0) = Region{2}(_region_ellipse((radius, radius)))

ellipse(radii = (10.0, 15.0)) = Region{2}(_region_ellipse(radii))

square(side = 10.0) = Region{2}(_region_rectangle((side, side)))

rectangle(sides = (10.0, 15.0)) = Region{2}(_region_rectangle(sides))

sphere(radius = 10.0) = Region{3}(_region_ellipsoid((radius, radius, radius)))

spheroid(radii = (10.0, 15.0, 20.0)) = Region{3}(_region_ellipsoid(radii))

cube(side = 10.0) = Region{3}(_region_cuboid((side, side, side)))

cuboid(sides = (10.0, 15.0, 20.0)) = Region{3}(_region_cuboid(sides))

function _region_ellipse((rx, ry))
    return r -> (r[1]/rx)^2 + (r[2]/ry)^2 <= 1 + extended_eps(Float64)
end

function _region_rectangle((lx, ly))
    return r -> abs(2*r[1]) <= lx * (1 + extended_eps()) &&
                abs(2*r[2]) <= ly * (1 + extended_eps())
end

function _region_ellipsoid((rx, ry, rz))
    return r -> (r[1]/rx)^2 + (r[2]/ry)^2 + (r[3]/rz)^2 <= 1 + eps()
end

function _region_cuboid((lx, ly, lz))
    return r -> abs(2*r[1]) <= lx * (1 + extended_eps()) &&
                abs(2*r[2]) <= ly * (1 + extended_eps()) &&
                abs(2*r[3]) <= lz * (1 + extended_eps())
end

end # module