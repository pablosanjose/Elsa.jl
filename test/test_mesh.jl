module MeshTest

using Test
using Elsa, SparseArrays, LinearAlgebra
using Elsa: nvertices, nedges, nsimplices

@test begin
    mesh = marchingmesh(2, 3)
    nvertices(mesh) == 6 && nedges(mesh) == 9 && nsimplices(mesh) == 4
end

end # module
