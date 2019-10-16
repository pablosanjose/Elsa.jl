ENV["OMP_NUM_THREADS"] = 4

export PardisoShift

struct PardisoShift
    point::Float64
    verbose::Bool
end

"""
    PardisoShift(p::Number; verbose = false)

When used as the `point = PardisoShift(p)` keyword argument in `diagonalize`, it forces use 
of the MKL Pardiso library for the shift-and-invert Lanczos method (it should be installed). 
`verbose = true` makes Pardiso verbose.

# Example
```jldoctest
julia> using Pardiso, Arpack; h = rand(10,10); h = h' + h;

julia> d = diagonalizer(h, Arpack_IRAM, point = PardisoShift(0.0))
Diagonaliser{Arpack_IRAM{Float64}} for (10, 10) Hermitian matrix around point 0.0
```
"""
PardisoShift(p::Number; verbose = false) = PardisoShift(p, verbose)

getpoint(p::PardisoShift) = p.point

function linearmap(h::AbstractArray{Tv}, p::PardisoShift) where {Tv}
    ps = pardisosolver(;verbose = p.verbose)
    _set_matrixtype!(ps, Tv)
    hp = ensuresparse(Pardiso.get_matrix(ps, h - Tv(p.point) * I, :N))
    preparesolver!(ps, hp)
    lmap = let ps = ps, hp = hp
        LinearMap{Tv}((x, b) -> Pardiso.pardiso(ps, x, hp, b), size(h)...,
                      ismutating = true, ishermitian = true)
    end
    return lmap, ps
end

function pardisosolver(; verbose = false)
    ps = Pardiso.MKLPardisoSolver()
    finalizer(releasePardiso, ps)
    verbose && Pardiso.set_msglvl!(ps, Pardiso.MESSAGE_LEVEL_ON)
    Pardiso.pardisoinit(ps)
    Pardiso.set_iparm!(ps, 12, 2) # Pardiso expects CSR, Julia uses CSC
    return ps
end

_set_matrixtype!(ps, ::Type{<:AbstractFloat}) = Pardiso.set_matrixtype!(ps, Pardiso.REAL_SYM_INDEF)
_set_matrixtype!(ps, ::Type{<:Complex}) = Pardiso.set_matrixtype!(ps, Pardiso.COMPLEX_HERM_INDEF)

ensuresparse(a::SparseMatrixCSC) = a
ensuresparse(a::AbstractMatrix) = sparse(a)

function preparesolver!(ps, hp::AbstractArray{Tv}) where {Tv}
    b = Vector{Tv}(undef, size(hp, 1))
    Pardiso.set_phase!(ps, Pardiso.ANALYSIS_NUM_FACT)
    Pardiso.pardiso(ps, hp, b)
    Pardiso.set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)
    return ps
end

function releasePardiso(ps)
    original_phase = Pardiso.get_phase(ps)
    Pardiso.set_phase!(ps, Pardiso.RELEASE_ALL);
    Pardiso.pardiso(ps)
    Pardiso.set_phase!(ps, original_phase)
    return ps
end