ENV["OMP_NUM_THREADS"] = 4
using Pardiso
using Elsa, BenchmarkTools, Arpack, LinearAlgebra, LinearMaps, KrylovKit, ArnoldiMethod, 
      IterativeSolvers, SuiteSparse

function solver()
    ps = MKLPardisoSolver()
    set_matrixtype!(ps, Pardiso.COMPLEX_HERM_INDEF)
    # set_msglvl!(ps, Pardiso.MESSAGE_LEVEL_ON)
    pardisoinit(ps)
    return ps
end
function lmap(h::AbstractMatrix{Tv}) where {Tv}
    ps = solver()
    A_pardiso = get_matrix(ps, h, :C);
    cols, rows = size(h)
    # b = zeros(eltype(h), cols)
    b = Vector(h[:,3])
    set_phase!(ps, Pardiso.ANALYSIS)
    # set_perm!(ps, randperm(n))
    pardiso(ps, A_pardiso, b)
    set_phase!(ps, Pardiso.NUM_FACT)
    pardiso(ps, A_pardiso, b)
    set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)
    LinearMap{Tv}((x, y) -> pardiso(ps, x, A_pardiso, y), cols, rows,
        ismutating = true, ishermitian = true)
end

function lmap2(h::AbstractMatrix{Tv}) where {Tv}
    cols, rows = size(h)
    # F = ldlt(h)
    F = lu(h)
    LinearMap{Tv}((x, y) -> ldiv!(x, F, y), cols, rows,
        ismutating = true, ishermitian = true)
end

sortreal(list, n = min(length(list), 10); rev = false) = sort(sort(list, by = abs, rev = rev)[1:n], by = real)

# sys = System(:honeycomb, Model(Hopping(1))) |> grow(region = Region(:circle, 300))
sys = System(:honeycomb, Model(Onsite(.2), Hopping(1, range = 1/√3))) |> grow(supercell = 30)
h = hamiltonian(sys, k = (0.21321,0.234213));
@time l = lmap(h);
@time l2 = lmap2(h);
lit = LOBPCGIterator(l, I, true, rand(eltype(h), size(h,1), 10));
lit2 = LOBPCGIterator(l2, I, true, rand(eltype(h), size(h,1), 10));

lob(lit) = sortreal(1 ./ lobpcg!(lit).λ);
arp(h) = sortreal(eigs(h, nev = 10, sigma = .01im)[1]);
ame(l) = sortreal(1 ./ partialschur(l, nev = 10)[1].eigenvalues);
kry(l) = sortreal(1 ./ eigsolve(x -> l * x, complex.(rand(size(l,1))), 10)[1]);

@btime lob($lit)
@btime arp($h)
@btime ame($l)
@btime kry($l)

@btime lob($lit2)
@btime arp($h)
@btime ame($l2)
@btime kry($l2)