function plot(lat::Lattice{T,3}) where T
    scene = Scene()
    
    for sublat in lat.sublats
        meshscatter!(scene, Point3f0.(sublat.sites))
    end

    cam3d!(scene)
    
    return scene
end
