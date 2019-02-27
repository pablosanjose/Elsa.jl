### Introduction

Green functions will be built from generic `Spectrum`s or from `System`s. In the latter case, if the system is 1D, we can resort to traditional generalised eigenvalue methods. Also, iterative matrix-free methods could be invoked. But in the case of getting a `Spectrum` we drop to the much more powerful interpolated methods. This issue describes some design ideas for the latter.

A `Spectrum` is essentially a `Mesh` and a `state` at each mesh node. The `N`-dimensional nodes `(parameters..., energy)` are part of a `Lattice`, that together with a collection of `Elements` (`N`-simplices of nodes) form the `Mesh`. The `parameters` are typically Bloch momenta (periodic), but can be any other system parameter (periodic or otherwise) that are used for interpolation of energies and states.
```julia
struct Mesh{T,E,L,N,EL}
    lattice::Lattice{T,E,L,EL}
    elements::Elements{N}
end
struct Spectrum{T,N,L,NL}
    bands::Mesh{T,N,L,N,NL}
    states::Matrix{Complex{T}}
end
```

To evaluate the Green function at a given energy we employ the analytic formulae for the integral of the resolvent in linear simplices. Now, the integral is performed *only* on the Bloch momenta, so we need a way to enconde which of the `N` dimensions of `lattice` elements we have to integrate over. One way is to have that be part of the `Spectrum`, but it's perhaps more flexible to pass that as a parameter to `GreenFunction(bands, blochdims = (1,2,3))`, where `blochdims` are the dimension to integrate over.

Should we make GreenFunction a type with some precomputation, and then compute its value at different energies and values of non-intergrated dimensions? What valuable precomputation can be performed?

### Slicing problem

Take a `N`-simplex in `N`-dimensional space. Its contribution to the Green function requires to perform a section at a given energy + non-Bloch parameters, `M` linear constraints in total. That should produce a set of vertices, which form a new "section simplex".

The original simplex is defined by the `N` vertices `p_1, p_2... p_N`. Taking `p0` as a reference vertex (any of them will do), we can define a matrix `v` with columns given by edge vectors containing `p0`, i.e. `v_ij = p^i_j - p0^i`
The simplex is then defined as all `N`-dimensional points satisfying
```
r_α = p0 + v *  α
```
where ` α` is an `N-1`-dimensional vector with `Σ|α_i| ∈ [0,1]`.

Any point `r` is written in a basis where the Bloch momenta `q` are the first `B` coordinates of `r`, and the non-Bloch + energy `μ` are the last 'M' coordinates (with `N = B + M`). We choose energy, which is always present, to be the first coordinate of `μ`, i.e. `r_(B+1) = ε`.
```
r = (q_1 … q_B ε μ_2 … μ_B)
```

We now wish to impose `M` constraints on the `μ` coordinates of the points in the simplex. The values of `μ` live in an `M` dimensional subspace with projector `P`, whose `MxN` matrix reads
```
P = hcat(zero(SMatrix{M,B,Int}), one(SMatrix{M,M,Int}))
```

In terms of this `P`, the points `r'_α` in the simplex slice satisfy
```
α_i > 0 and ∑α_i <= 1       (constraint 1)
P * p0 + P * v * α = μ      (constraint 2)
r'_α = p0 + v *  α
```
The new constraint 2 (which are actually `M` equations on the acceptable values of `α`) leaves `B` degrees of freedom in `α`, with the rest `M` of them fixed. The simplex slice can then be decomposed as a set of `B`-simplices with vertices to be determined as follows.

Of the `N` values of `α_i` we select subsets of size `B` (there are `K = binomial(N, B)` of these subsets) and fix the corresponding `α_i` to zero in constraint 2, which then becomes a set of `M` equations with `M` unknowns. If it is invertible and the solution satisfies constraint 1 (`β_i > 0` and `∑β_i <= 1`), one has a vertex from which to build the new `B`-simplices that make up the slice. We call the `k` solutions `β^j` (vectors of length `M`), where `j=1…k` (`k <= K`, at most the number of subsets), and the actual vertices `p'^j = p0 + v * α^j`, where the `α^j` are the `N`-vectors built from each of the `β^j` `M`-vectors.

Can we find a closed form for `β^j`? Let us define the `K` projectors `Q^j` of size `MxN` such that `β = Q^j * α` selects the `M` non-zero components of `α` for a subset j, and conversely `α = transpose(Q^j) * β` builds an `N`-vector with `B` zeroes. As an example, 
```
Q^1 = hcat(zero(SMatrix{M,B,Int}), one(SMatrix{M,M,Int}))
```
selects the last `M` components of `α`. Constraint 2 is then written as
```
P * p0 + P * v * transpose(Q^j) * β = μ 
```
Then a candidate solution `β^j` would be expressed as
```
β^j = inv(A) * (μ - μ0)
A^j = P * v * transpose(Q^j)
μ0 = P * p0
```
This will be a valid solution as long as the `MxM` matrix `A^j` is invertible and satisfies constraint 1 (`β_i > 0` and `∑β_i <= 1`). Otherwise we discard the solution.

It is then clear that the `μ`-independent preprocessing that can be performed on a given mesh of simplices is to compute the inverses of matrices `A^j` for each subset and each simplex (where it exists), and store all the `μ0`'s.

### Pure-energy slices

In some cases we want to keep solutions even if we violate constraint 1. This happens when expressing the contribution of a simplex to the Green's function integral, that is expressed in terms of all the `β^j`, regardless of constraint 1. In such situation we need to slice **first** for `μ_2…μ_B` using constraint 1, create a new simplex mesh, and **then** use the same machinery to obtain the `p'` vertices when constraining `ɛ` without contraint 1 (no need to build a new mesh for that).

For other uses, such as computing a Fermi line, we do need to apply constraint 1 to find he Fermi line segment on each simplex.

Note that a pure-energy slice of an `N = B + 1`-dimensional mesh creates `K = binomial(N, N-1) = N` vertices per simplex, before constraint 1. Note that after contraint 1 we can have three different situation:

- (1) If the number `k` of constrained solutions is equal to `B = N - 1`, then there is a single slice `B`-simplex per original simplex. 

- (2) If all solutions remain valid (`k = K = N`) there will be more than one simplex. 

- (3) There can be no solutions (`k = 0`). 

I don't think you can get `0 < k < B`. 

In a 2D system (`N = 3`), you always have case (1), i.e. `k = 2`, single 2-simplex, a single Fermi line segment per simplex, or (3). Same for a 1D system. In a 3D system (`N = 4`), however, you can have any of the three cases.