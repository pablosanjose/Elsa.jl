using Makie
import Makie: plot

const colorscheme = Iterators.cycle(map(t -> RGBAf0(t...), ((0.410,0.067,0.031),(0.860,0.400,0.027),(0.940,0.780,0.000),(0.640,0.760,0.900),(0.310,0.370,0.650),(0.600,0.550,0.810),(0.150,0.051,0.100),(0.870,0.530,0.640),(0.720,0.130,0.250))))

function darken(rgba::T, v=0.3) where T
    r = max(0, min(rgba.r - v, 1))
    g = max(0, min(rgba.g - v, 1))
    b = max(0, min(rgba.b - v, 1))
    T(r,g,b,rgba.alpha)
end
function lighten(c, v=0.3)
    darken(c, -v)
end

function plot(lat::Lattice; resolution = (1024, 1024), siteradius = 0.2, siteborder = 8, quality = :low)
    scene = Scene(resolution = resolution)
    colors = collect(Iterators.take(colorscheme, nsublats(lat)))

    for (sublat, color) in zip(lat.sublats, colors)
        sites = sublat.sites
        if quality == :low
            scatter!(scene, sites, 
                strokewidth = siteborder, markersize = 2siteradius, color = color, strokecolor = darken(color))
        else
            meshscatter!(scene, sites, 
                strokewidth = siteborder, markersize = siteradius, color = color, strokecolor = darken(color))
        end
    end

    for ci in CartesianIndices(lat.links.intralink.slinks)
        i, j = Tuple(ci)
        col1, col2 = colors[i], colors[j]
        slink = lat.links.intralink.slinks[ci]
        for (r, dr) in slink.rdr
            links = [Point(r) => Point(r + dr/2) for (r, dr) in slink.rdr]
            linesegments!(links, color = col1)
        end
        # links = [Point3f0(r) => Point3f0(r + dr/2) for (r, dr) in slink.rdr]
        # linesegments!(links, color = col1)
        # linesegments!([Point3f0(r) => Point3f0(r - dr/2) for (r, dr) in slink.rdr], color = col2)
    end

    cam3d!(scene)
    scale!(scene)
    center!(scene)
    
    return scene
end
