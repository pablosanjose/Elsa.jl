struct Preset{S<:NamedTuple}
    name::Symbol
    kwargs::S
end

Preset(name; kwargs...) = Preset(name, kwargs.data)

#######################################################################
# Lattice presets
#######################################################################

latticepresets = Dict(
    :linear => () ->
        Lattice(Sublat((0.,)), Bravais((1.,))),
    :square => () ->
        Lattice(Sublat((0., 0.)), Bravais((1., 0.),(0., 1.))),
    :triangular => () ->
        Lattice(Sublat((0.,0.)), Bravais((cos(pi/3), sin(pi/3)),(-cos(pi/3), sin(pi/3)))),
    :honeycomb => () ->
        Lattice(Sublat((0.0, -0.5/sqrt(3.0)); name = :A),
        Sublat((0.0, 0.5/sqrt(3.0)); name = :B), Bravais((cos(pi/3), sin(pi/3)), (-cos(pi/3), sin(pi/3)))),
    :graphene => () ->
        Lattice(Sublat((0.0, -0.5/sqrt(3.0)); name = :A),
        Sublat((0.0, 0.5/sqrt(3.0)); name = :B), Bravais((cos(pi/3), sin(pi/3)), (-cos(pi/3), sin(pi/3))),
        LinkRule(1/√3), LatticeConstant(0.246)),
    :cubic => () ->
        Lattice(Sublat((0., 0., 0.)), Bravais((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))),
    :fcc => () ->
        Lattice(Sublat((0., 0., 0.)), Bravais(@SMatrix([-1. -1. 0.; 1. -1. 0.; 0. 1. -1.])'/sqrt(2.))),
    :bcc => () ->
        Lattice(Sublat((0., 0., 0.)), Bravais((1., 0., 0.), (0., 1., 0.), (0.5, 0.5, 0.5))),
    :honeycomb_bilayer =>
        function (;twistindex = 1, twistindices = (twistindex, 0), interlayerdistance = 1.0, linkrangeintralayer = 1/sqrt(3))
            (m, r) = twistindices
            θ = acos((3m^2 + 3m*r +r^2/2)/(3m^2 + 3m*r + r^2))
            sAbot = Sublat((0.0, -0.5/sqrt(3.0), - interlayerdistance / 2); name = :Ab)
            sBbot = Sublat((0.0,  0.5/sqrt(3.0), - interlayerdistance / 2); name = :Bb)
            sAtop = Sublat((0.0, -0.5/sqrt(3.0),   interlayerdistance / 2); name = :At)
            sBtop = Sublat((0.0,  0.5/sqrt(3.0),   interlayerdistance / 2); name = :Bt)
            bravais = Bravais((cos(pi/3), sin(pi/3)), (-cos(pi/3), sin(pi/3)))
            if gcd(r, 3) == 1
                scbot, sctop = @SMatrix[m -(m+r); (m+r) 2m+r], @SMatrix[m+r -m; m 2m+r]
            else
                scbot, sctop = @SMatrix[m+r÷3 -r÷3; r÷3 m+2r÷3], @SMatrix[m+2r÷3 r÷3; -r÷3 m+r÷3]
            end
            ltop = Lattice(sAtop, sBtop, bravais, Dim(3), LinkRule(linkrangeintralayer), Supercell(sctop))
            lbot = Lattice(sAbot, sBbot, bravais, Dim(3), LinkRule(linkrangeintralayer), Supercell(scbot))
            let R = @SMatrix[cos(θ/2) -sin(θ/2) 0; sin(θ/2) cos(θ/2) 0; 0 0 1]
                transform!(ltop, r -> R * r)
            end
            let R = @SMatrix[cos(θ/2) sin(θ/2) 0; -sin(θ/2) cos(θ/2) 0; 0 0 1]
                transform!(lbot, r -> R * r)
            end
            combine_nocopy(lbot, ltop)
        end
    )

regionpresets = Dict(
    :circle => (radius = 10.0, ; kw...) -> Region{2}(_region_ellipse((radius, radius)); kw...),
    :ellipse => (radii = (10.0, 15.0), ; kw...) -> Region{2}(_region_ellipse(radii); kw...),
    :square => (side = s, ; kw...) -> Region{2}(_region_rectangle((side, side)); kw...),
    :rectangle => (sides = (10.0, 15.0), ; kw...) -> Region{2}(_region_ellipsoid((radius, radius, radius)); kw...),
    :sphere => (radius = 10.0, ; kw...) -> Region{3}(_region_ellipsoid((radius, radius, radius)); kw...),
    :spheroid => (radii = (10.0, 15.0, 20.0), ; kw...) -> Region{3}(_region_ellipsoid(radii); kw...),
    :cube => (side = 10.0, ; kw...) -> Region{3}(_region_cuboid((side, side, side)); kw...),
    :cuboid => (sides = (10.0, 15.0, 20.0), ; kw...) -> Region{3}(_region_cuboid(sides); kw...)
    )


function _region_ellipse(radii)
    return r -> (r[1]/radii[1])^2 + (r[2]/radii[2])^2 <= 1 + extended_eps()
end

function _region_rectangle(sides)
    return r -> abs(2*r[1])<= sides[1] * (1 + extended_eps()) &&
                abs(2*r[2])<= sides[2] * (1 + extended_eps())
end

function _region_ellipsoid(radii)
    return r -> (r[1]/radii[1])^2 + (r[2]/radii[2])^2 + (r[3]/radii[3])^2 <= 1 + eps()
end

function _region_cuboid(sides)
    return r -> abs(2*r[1])<=sides[1] * (1 + extended_eps()) &&
                abs(2*r[2])<=sides[2] * (1 + extended_eps()) &&
                abs(2*r[3])<=sides[3] * (1 + extended_eps())
end

#######################################################################
# Model presets
#######################################################################

modelpresets = Dict(
    :kinetic2D => ((; mass = 1, a0 = 1) -> Model(
            Onsite(r-> (2.0 ./ (mass.*a0^2)) .* eye(SMatrix{2,2, Float64})),
            Hopping((r,dr)-> (-1.0 ./ (mass.*a0^2)) .* eye(SMatrix{2,2, Float64}))))
    )
