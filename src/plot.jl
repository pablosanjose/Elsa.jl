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
function lighten(rgba, v = 0.66)
    darken(rgba, -v)
end
transparent(rgba::T, v = 0.5) where T = T(rgba.r, rgba.g, rgba.b, rgba.alpha * v) 

 function plot(lat::Lattice; resolution = (1024, 1024), kwargs...)
    scene = Scene(resolution = resolution)
    colors = collect(take(colorscheme, nsublats(lat)))
    
    plotcell!(scene, lat, lat.links.intralink, colors; dimming = 0, kwargs...)
    for ilink in lat.links.interlinks
        plotcell!(scene, lat, ilink, colors; kwargs...)
    end

    b1, b2 = boundingboxlat(lat)
    lookat = (b1 + b2)/2
    eye = lookat + SVector(0.,-.1,2.)*norm((b1 - b2)[1:2])

    cam3d!(scene)
    scale!(scene)
    update_cam!(scene, Vec3f0(eye), Vec3f0(lookat), Vec3f0(0,1,0))

    return scene
 end

 function plotcell!(scene, lat, ilink, colors; shaded = false, dimming = 0.75, kwargs...)
    celldist = bravaismatrix(lat) * ilink.ndist
    for (sublat, color) in zip(lat.sublats, colors)
        colordimmed = transparent(color, 1 - dimming)
        sites = [Point3f0(celldist + site) for site in sublat.sites]
        shaded ? drawsites_hi!(scene, sites, colordimmed; kwargs...) : drawsites_lo!(scene, sites, colordimmed; kwargs...)
    end

    for ci in CartesianIndices(ilink.slinks)
        i, j = Tuple(ci)
        col1, col2 = darken(colors[j], 0.1), darken(colors[i], 0.1)
        col2 = transparent(col2, 1 - dimming)
        slink = ilink.slinks[ci]
        shaded ? 
            drawlinks_hi!(scene, slink.rdr, (col1, col2); kwargs...) : 
            drawlinks_lo!(scene, slink.rdr, (col1, col2); kwargs...)
    end
    return nothing
end

function drawsites_lo!(scene, sites, color; siteradius = 0.2, strokewidth = 15)
    isempty(sites) || scatter!(scene, sites, 
        markersize = 2siteradius, color = color, strokewidth = strokewidth, strokecolor = darken(color, 0.3))
    return nothing
end

function drawsites_hi!(scene, sites, color; siteradius = 0.2)
    isempty(sites) || meshscatter!(scene, sites, markersize = siteradius, color = color)
    return nothing
end

function drawlinks_lo!(scene, rdr, (col1, col2); siteradius = 0.2, strokewidth = 15)
    isempty(rdr) && return nothing
    segments = [fullsegment(r, dr, siteradius * 0.99) for (r, dr) in rdr]
    colsegments = collect(take(cycle((col1, col2)), 2 * length(segments)))
    linesegments!(scene, segments, linewidth = strokewidth, color = colsegments)
    return nothing
end

function drawlinks_hi!(scene, rdr, (col1, col2); siteradius = 0.2, linkradius = 0.1)
    isempty(rdr) && return nothing
    # positions = view(rdr, :, 1)
    positions = [Point3f0(r) for (r, _) in rdr]
    segments = [halfsegment(r, dr, 0) for (r, dr) in rdr]
    scales = [Vec3f0(linkradius, linkradius, norm(dr)) for dr in segments]
    cylinder = GLNormalMesh(Makie.Cylinder{3, Float32}(Point3f0(0., 0., 0.), Point3f0(0., 0, 1.), Float32(1)), 12)
    meshscatter!(scene, positions, marker = cylinder, markersize = scales, rotations = segments, color = col2)
    cylinder = GLNormalMesh(Makie.Cylinder{3, Float32}(Point3f0(0., 0., 0.), Point3f0(0., 0, -1.), Float32(1)), 12)
    meshscatter!(scene, positions,  marker = cylinder, markersize = scales, rotations = segments, color = col1)
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