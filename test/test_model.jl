module ModelTest

using Test
using Elsa: TighbindingModel, OnsiteTerm, HoppingTerm

@testset "onsite" begin
    r = SVector(0.0, 0.0)
    for o in (1, 1.0, @SMatrix[1 0; 0 1], I, r -> 1, r -> I, r -> SMatrix{3,3}(I))
        for s in (missing, :A, (:A,), (:A, :B)), c in (1, 1.0, 1.0f0)
            ons = c * onsite(o, sublats = s)
            @test ons isa
                Elsa.TightbindingModel{1,Tuple{Elsa.OnsiteTerm{typeof(o),S,typeof(c)}}} where S
            term = first(ons.terms)
            for t in (Float32, Float64, SMatrix{3,3}, SMatrix{3,2,Float64}, SMatrix{1,4,Float32})
                @test Elsa.padtotype(term(r, r), t) isa t
            end
        end
    end
end

end # module


# @testset "model terms" begin
#     for o in (1, @SMatrix[1], @SMatrix[1 2; 3 4.0], r -> 2, r -> @SMatrix[r[1] 0; 0 r[2]]),
#         sl in (missing, 1, (1,), (2,1), (2, :A))
#         @test Onsite(o, sublats = sl) isa Elsa.Onsite{typeof(o)}
#     end
#     @test Onsite(1, sublats = (2, :A)).sublats == ((2, 2), (:A, :A))
#     @test_throws DimensionMismatch Onsite(@SMatrix[1 2;], sublats = (2,:A))

#     testhoppings(o, sl, nd, r) = if o isa Function
#         @test Hopping(o, sublats = sl, ndists = nd, range = r) isa Elsa.Hopping{typeof(o)}
#     else
#         @test Hopping(o, sublats = sl, ndists = nd, range = r) isa Elsa.Hopping{<:SMatrix}
#     end

#     for o in (1, @SMatrix[1], @SMatrix[1 2; 3 4.0], @SMatrix[1 2;], r -> 2,
#               r -> @SMatrix[r[1] 0; 0 r[2]]),
#         sl in (missing, 1, ((1, 1),), ((1, 1), (1, 2))),
#         nd in (missing, (0, 0), ((0, 1), (2, 2))),
#         r in (1.1, 2f0)
#         testhoppings(o, sl, nd, r)
#     end
# end

# @testset "model" begin
#     @test isempty(Model().terms)
#     @test_throws DimensionMismatch System(LatticePresets.honeycomb(),
#         Model(Hopping(@SMatrix[1 2], sublats = 2)))
#     @test System(LatticePresets.honeycomb(),
#         Model(Hopping(@SMatrix[1 2], sublats = (1, 2)))) isa System
# end

