using Makie, GeometryTypes
import AbstractPlotting: default_theme, Plot, plot!, to_value
using Base.Iterators: take, cycle

function default_theme(scene::SceneLike, ::Type{<: Plot(Lattice)})
    Theme(
        resolution = (1024, 1024), allintra = false, allcells = true, intralinks = true, interlinks = true,
        shaded = false, dimming = 0.75, 
        siteradius = 0.2, siteborder = 20, 
        linkthickness = 15, linkoffset = 0.99, linkradius = 0.1,
        colorscheme = map(t -> RGBAf0(t...), ((0.410,0.067,0.031),(0.860,0.400,0.027),(0.940,0.780,0.000),(0.640,0.760,0.900),(0.310,0.370,0.650),(0.600,0.550,0.810),(0.150,0.051,0.100),(0.870,0.530,0.640),(0.720,0.130,0.250)))
        )
end

function AbstractPlotting.plot!(plot::Plot(Lattice))
    lat = to_value(plot[1])
    colors = collect(take(cycle(plot[:colorscheme]), nsublats(lat)))
    
    scene = Scene(resolution = plot[:resolution])
    cam3d!(scene)
    scale!(scene)
    
    celldist0 = bravaismatrix(lat) * lat.links.intralink.ndist
    for ilink in lat.links.interlinks
        celldist = bravaismatrix(lat) * ilink.ndist
        plot[:allintra] && plotlinks!(scene, lat.links.intralink, celldist, plot; dimming = plot[:dimming])
        plot[:interlinks] && plotlinks!(scene, ilink, celldist0, colors, plot; dimming = plot[:dimming])
        plot[:allcells] && plotsites!(scene, lat, celldist, colors, plot; dimming = plot[:dimming])
    end
    intralinks && plotlinks!(scene, lat.links.intralink, celldist0, colors, plot; dimming = 0.0)
    plotsites!(scene, lat, celldist0, colors, plot; dimming = 0.0)

    b1, b2 = boundingboxlat(lat)
    lookat = Vec3D((b1 + b2)/2)
    eye = lookat + Vec3f0(0.,0.,2.)*normxy(b1 - b2)

    update_cam!(scene, eye, lookat, Vec3f0(0,1,0))

    return scene
 end


 function plotsites!(scene, lat, celldist, colors, plot; dimming = 0.0)
    for (sublat, color) in zip(lat.sublats, colors)
        colordimmed = transparent(color, 1 - dimming)
        sites = [Point3D(celldist + site) for site in sublat.sites]
        plot[:shaded] ? drawsites_hi!(scene, sites, colordimmed, plot) : drawsites_lo!(scene, sites, colordimmed, plot)
    end
end

function plotlinks!(scene, ilink, celldist, colors, plot; dimming = 0.0)
    for ci in CartesianIndices(ilink.slinks)
        i, j = Tuple(ci)
        col1, col2 = darken(colors[j], 0.1), darken(colors[i], 0.1)
        col2 = transparent(col2, 1 - dimming)
        iszero(celldist) || (col1 = transparent(col1, 1 - dimming))
        slink = ilink.slinks[ci]
        plot[:shaded] ? 
            drawlinks_hi!(scene, slink.rdr, celldist, (col1, col2); kwargs...) : 
            drawlinks_lo!(scene, slink.rdr, celldist, (col1, col2); kwargs...)
    end
    return nothing
end

function drawsites_lo!(scene, sites, color, plot)
    isempty(sites) || scatter!(scene, sites, 
        markersize = 2*plot[:siteradius], color = color, strokewidth = plot[:siteborder],  strokecolor = darken(color, 1.0))
    return nothing
end

function drawsites_hi!(scene, sites, color, plot)
    isempty(sites) || meshscatter!(scene, sites, markersize = plot[:siteradius], color = color)
    return nothing
end

function drawlinks_lo!(scene, rdr, celldist, (col1, col2), plot)
    isempty(rdr) && return nothing
    segments = [fullsegment(celldist + r, dr, plot[:siteradius] * plot[:linkoffset]) for (r, dr) in rdr]
    colsegments = collect(take(cycle((col1, col2)), 2 * length(segments)))
    linesegments!(scene, segments, linewidth = plot[:linkthickness], color = colsegments)
    return nothing
end

function drawlinks_hi!(scene, rdr, celldist, (col1, col2), plot)
    isempty(rdr) && return nothing
    # positions = view(rdr, :, 1)
    positions = [Point3f0(celldist + r) for (r, _) in rdr]
    segments = [halfsegment(r, dr, 0) for (r, dr) in rdr]
    scales = [Vec3f0(plot[:linkradius], plot[:linkradius], norm(dr)) for dr in segments]
    cylinder = GLNormalMesh(Makie.Cylinder{3, Float32}(Point3f0(0., 0., 0.), Point3f0(0., 0, 1.), Float32(1)), 12)
    meshscatter!(scene, positions, marker = cylinder, markersize = scales, rotations = segments, color = col2)
    cylinder = GLNormalMesh(Makie.Cylinder{3, Float32}(Point3f0(0., 0., 0.), Point3f0(0., 0, -1.), Float32(1)), 12)
    meshscatter!(scene, positions,  marker = cylinder, markersize = scales, rotations = segments, color = col1)
    return nothing
end

function fullsegment(r, dr, rad) 
    dr2 = dr*(1 - 2rad/norm(dr))/2
    return Point3D(r - dr2) => Point3D(r + dr2)
end

function halfsegment(r, dr, rad) 
    dr2 = dr*(1 - 2rad/norm(dr))/2
    return  Vec3D(dr2)
end


Point3D(r::SVector{3,T}) where T = Point3f0(r)
Point3D(r::SVector{N,T}) where {N,T} = Point3f0(padright(r, zero(Float32), Val(3)))
Vec3D(r::SVector{3,T}) where T = Vec3f0(r)
Vec3D(r::SVector{N,T}) where {N,T} = Vec3f0(padright(r, zero(Float32), Val(3)))

normxy(sv::SVector{3}) = norm(sv[1:2])
normxy(sv) = norm(sv)

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