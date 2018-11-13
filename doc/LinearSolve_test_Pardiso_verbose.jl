ENV["OMP_NUM_THREADS"] = 4
const SIZE = 30000

using Pardiso
using Random
using SparseArrays
using LinearAlgebra
using BenchmarkTools
using Printf

Random.seed!(1);

ps = MKLPardisoSolver()
pardisoinit(ps)
# set_matrixtype!(ps, 13)
set_matrixtype!(ps, Pardiso.COMPLEX_HERM_POSDEF)
set_msglvl!(ps, Pardiso.MESSAGE_LEVEL_ON)

b = rand(Complex{Float64}, SIZE);
A0 = sprand(Complex{Float64}, SIZE, SIZE, 2.0/SIZE);
A = A0 + A0' + spdiagm(0 => 0.1 .* b);
A_pardiso = get_matrix(ps, A, :N);

# Analyze the matrix and compute a symbolic factorization.
set_phase!(ps, Pardiso.ANALYSIS)
pardiso(ps, A_pardiso, b)
@printf("The factors have %d nonzero entries.\n", get_iparm(ps, 18))

# Compute the numeric factorization.
set_phase!(ps, Pardiso.NUM_FACT)
pardiso(ps, A_pardiso, b)

# Compute the solutions X using the symbolic factorization.
set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)
x = similar(b); # Solution is stored in X
@time pardiso(ps, x, A_pardiso, b)
@printf("PARDISO performed %d iterative refinement steps.\n", get_iparm(ps, 7))

# Compute the residuals.
r = norm(A*x - b)/norm(b)
@printf("The maximum residual for the solution is %0.3g.\n",maximum(r))

# Free the PARDISO data structures.
set_phase!(ps, Pardiso.RELEASE_ALL)
pardiso(ps)
