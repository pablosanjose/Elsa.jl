ENV["OMP_NUM_THREADS"] = 4
const SIZE = 30000

using Pardiso
using Random
using SparseArrays
using LinearAlgebra
using BenchmarkTools

Random.seed!(1);

ps = MKLPardisoSolver()
pardisoinit(ps)
# set_matrixtype!(ps, 13)
set_matrixtype!(ps, -4)
set_msglvl!(ps, Pardiso.MESSAGE_LEVEL_ON)
b = rand(Complex{Float64}, SIZE);
# b = zeros[Complex{Float64}, SIZE]; b[1] = 1.0 + 0im;
A0 = sprand(Complex{Float64}, SIZE, SIZE, 2.0/SIZE);
A = A0 + A0' + spdiagm(0 => 0.1 .* real.(b));
x = similar(b);
# @time X = solve(ps, A, B);
@time solve!(ps, x, A, b);
rel_err = norm(A*x - b) / norm(b);

(A[1,1], b[1])
@show rel_err;
x[1]
