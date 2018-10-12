using Makie, GeometryTypes
import Makie: plot
using Base.Iterators: take, cycle

const colorscheme = cycle(map(t -> RGBAf0(t...), ((0.410,0.067,0.031),(0.860,0.400,0.027),(0.940,0.780,0.000),(0.640,0.760,0.900),(0.310,0.370,0.650),(0.600,0.550,0.810),(0.150,0.051,0.100),(0.870,0.530,0.640),(0.720,0.130,0.250))))

function darken(rgba::T, v = 0.66) where T
    r = max(0, min(rgba.r * (1 - v), 1))
    g = max(0, min(rgba.g * (1 - v), 1))
    b = max(0, min(rgba.b * (1 - v), 1))
    T(r,g,b,rgba.alpha)
end
function lighten(c, v = 0.66)
    darken(c, -v)
end

# plot(lat::Lattice; highquality::Bool = false, kwargs...) = highquality ? _plothi(lat, kwargs...) : _plotlow(lat, kwargs...)
 function plot(lat::Lattice; highquality = false, resolution = (1024, 1024), kwargs...) #siteradius = 0.2, siteborder = 15)
    scene = Scene(resolution = resolution)
    colors = collect(take(colorscheme, nsublats(lat)))

    for (sublat, color) in zip(lat.sublats, colors)
        sites = sublat.sites
        highquality ? drawsites_hi!(scene, sites, color; kwargs...) : drawsites_lo!(scene, sites, color; kwargs...)
    end
        

    for ci in CartesianIndices(lat.links.intralink.slinks)
        i, j = Tuple(ci)
        col1, col2 = darken(colors[j], 0.1), darken(colors[i], 0.1)
        slink = lat.links.intralink.slinks[ci]
        highquality ? 
            drawlinks_hi!(scene, slink.rdr, (col1, col2); kwargs...) : 
            drawlinks_lo!(scene, slink.rdr, (col1, col2); kwargs...)
    end

    cam3d!(scene; eyeposition = Vec3f0(0, 0, 3))
    scale!(scene)
    center!(scene)
    
    return scene
end

function drawsites_lo!(scene, sites, color; siteradius = 0.2, siteborder = 15)
    isempty(sites) || scatter!(scene, sites, 
        strokewidth = siteborder, markersize = 2siteradius, color = color, strokecolor = darken(color, 0.3))
    return nothing
end

function drawsites_hi!(scene, sites, color; siteradius = 0.2)
    isempty(sites) || meshscatter!(scene, sites, 
        markersize = siteradius, color = color, strokecolor = darken(color, 0.3))
    return nothing
end

function drawlinks_lo!(scene, rdr, (col1, col2); siteradius = 0.2, siteborder = 15)
    segments = [fullsegment(r, dr, siteradius * 0.99) for (r, dr) in rdr]
    colsegments = collect(take(cycle((col1, col2)), 2 * length(segments)))
    isempty(segments) || linesegments!(scene, segments, linewidth = siteborder, color = colsegments)
    return nothing
end

function drawlinks_hi!(scene, rdr, (col1, col2); siteradius = 0.2, linkradius = 0.1)
    cylinder = GLNormalMesh(Makie.Cylinder{3, Float32}(Point3f0(0., 0., 0.), Point3f0(0., 0, 1.), Float32(1)), 12)
    # positions = view(rdr, :, 1)
    positions = [Point3f0(r) for (r, _) in rdr]
    segments = [halfsegment(r, dr, siteradius * 0.99) for (r, dr) in rdr]
    scales = [Vec3f0(linkradius, linkradius, norm(dr)) for dr in segments]
    if isempty(segments) 
        meshscatter!(scene, positions, markersize = scales, rotations = segments, color = col1)
        scales .*= -1f0
        meshscatter!(scene, positions, markersize = scales, rotations = segments, color = col2)
    end
    return nothing
end

function fullsegment(r, dr, rad) 
    dr2 = dr*(1 - 2rad/norm(dr))/2
    return Point3f0(r - dr2) => Point3f0(r + dr2)
end

function halfsegment(r, dr, rad) 
    dr2 = dr*(1 - 2rad/norm(dr))/2
    return  Vec3f0(dr2)
end