using Makie
import Makie: plot

function plot(lat::Lattice)
    scene = Scene(resolution = (1024, 1024))
    
    for sublat in lat.sublats
        meshscatter!(scene, Makie.Point3f0.(sublat.sites), scale = (1,1,1))
    end

    cam3d!(scene)
    scale!(scene)
    center!(scene)
    
    return scene
end
