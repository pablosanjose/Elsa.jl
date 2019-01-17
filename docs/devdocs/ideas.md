# Method space by system type

## Bounded systems (wrapped or open)
- Discrete spectra 
    - E.g. Majorana nanowires
- KPM methods (Pybinding level)
    - Computation of momenta `μ_mn… = ⟨ξ|Tm(H)Tn(H)…|ξ⟩`
    - Both local (DOS, σ...) and random (`LDOSᵣ`) `|ξ⟩`
    - Postprocess to obtain `DOS`, `LDOSᵣ`,`σ`, `σ_dc`

## Bloch systems (no disorder)
- Bandstructures in simplices
    - Recursive refinement?
    - Fermi surfaces
- Green functions (retarded + Keldysh)
    - Simplex integrals
- Matrix-free KPM methods (overkill?)
    - Same functionality as bounded but avoiding allocation of `H`
        - Parallel processing (ghosts, SIMD tiles)
    - For local `|ξ⟩` we can adjust its size with the order of the momenta expansion

## Bloch systems with boundaries
- Green functions
    - Method of images

## Bloch systems with bounded disorder
- `T` matrix from not-huge disorder `V` (≲ 10⁴ orbitals)
    - `LDOSᵣ(ω)` for fixed `r` by linear solving *vector* `TG0(_,r)` from `(1-VG0)TG0 = VG0`, and then `G(r,r) = G0(r,r) + G(r,_)TG0(_,r)`
    - This can be accelerated for other `r`'s by doing a factorization of `(1-VG0)` (dense) for a given `ω`, if `V` is not too large (~10⁴ orbitals in 5 seconds)
    - For larger ensembles, could use Krylov 
        - I don't know if you gain any speed, but you don't need to store a factorisation
- Matrix-free KPM methods (> 10⁴ orbitals)
    - Same as without disorder, but adding `V` disorder on-the-fly
    - Should rely on a package that takes care of parallelisation issues in this case (non-trivial)
    - Energy resolution is limited by expansion order -> system size
    - May assume periodic or simple open boundaries

## Bloch systems with unbounded disorder/perturbations
- Matrix-free KPM methods
    - Applying the perturbations on the fly. 
    - If the perturbation is random, ensure deterministic RNG in each loop
- Since the perturbation is both unbounded and non-periodic, Green function methods seem hopeless







# Method space by problem type

## Disorder averaged DOS/LDOS
The general strategy involves implementing a large enough system with a single disorder realisation so that the result is self-averaged. The more dilute the disorder, the larger the system needs to be to self-average.

We can apply these methods also to systems in which the disorder is 

- KPM
    - For really huge systems we can do matrix-free KPM. Otherwise we can store the sparse Hamiltonian (which doesn't make use of the system's locality)
    - Returns the DOS/LDOS across the full bandwidth
    - **Drawbacks**
        - Energy resolution is full bandwidth `W` divided by expansion order `O`. `O` should be smaller than linear system size (in unit cells) to probe true bulk properties ⇒ Difficult to reach sub-meV.
        - For DOS there is the additional problem of stochastic trace noise. Suppressed slowly at large system sizes.
- Green function
    - Requires solving the `T` matrix (order `N^2`), or better `Tr(G0TG0)` for the DOS, `⟨r|G0TG0|r⟩` for the LDOS. 
        - To do so we can build `|VG0ξ⟩=VG0|ξ⟩` with random or local `|ξ⟩` and `V` the disorder, and then solve `TG0|ξ⟩ = |TG0ξ⟩ = (1-VG0)^{-1}|VG0ξ⟩` by linear-solving the dense system `(1-VG0)|TG0ξ⟩ = |VG0ξ⟩` (can use matrix-free methods). Finally project onto `⟨ξ|G0`.
        - This takes a space equal to the size of the disorder, i.e. it is an order `V` method, much better than order `N` methods, but requires solving dense linear systems.
    - The energy resolution of the result is adaptive to each band: `Δϵ ∼ Wᵢ/Nk`, where `Wᵢ` is a subband width, and `Nk` the linear number of k-points.
    - **Drawbacks**
        - We need one simulation per energy. By the same token we only scan the energies we need.
        - It remains to be seen what is the bottleneck here, the calculation of `G0` (sum over simplices) or of `T` (solving the linear system).
    
