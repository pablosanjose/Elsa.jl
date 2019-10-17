#######################################################################
# Diagonalizer
#######################################################################
abstract type DiagonalizePackage end

struct Diagonalizer{M<:DiagonalizePackage,A<:AbstractArray,C}
    method::M
    matrix::A
    levels::Int
    origin::Float64
    minprojection::Float64
    codiag::C       # Matrices to resolve degeneracies, or missing
end

function Diagonalizer(method, matrix::AbstractMatrix{M};
             levels = missing,
             origin = 0.0,
             minprojection = 0.1,
             codiag = missing) where {M}
    _levels = levels === missing ? size(matrix, 1) : levels
    Diagonalizer(method, matrix, _levels, origin, minprojection, codiag)
end

# This is in general type unstable, a function barrier when using it is needed
diagonalizer(h::Hamiltonian{<:Lattice,<:Any,<:Any,<:Matrix}; kw...) =
    diagonalizer(LinearAlgebraPackage(values(kw)), similarmatrix(h))

function diagonalizer(h::Hamiltonian{<:Lattice,<:Any,M,<:SparseMatrixCSC};
                      levels = missing, origin = 0.0, codiag = missing, kw...) where {M}
    if size(h, 1) < 50 || levels === missing || levels / size(h, 1) > 0.1
        # @warn "Requesting significant number of sparse matrix eigenvalues. Converting to dense."
        matrix = Matrix(similarmatrix(h))
        _matrix = ishermitian(h) ? Hermitian(matrix) : matrix
        d = diagonalizer(LinearAlgebraPackage(; kw...), _matrix;
            levels = levels, origin = origin, codiag = codiag)
    elseif M isa Number
        matrix = similarmatrix(h)
        d = diagonalizer(ArpackPackage(; kw...), matrix;
            levels = levels, origin = origin, codiag = codiag)
    elseif M isa SMatrix
        matrix = similarmatrix(h)
        d = diagonalizer(ArnoldiPackagePackage(; kw...), matrix;
            levels = levels, origin = origin, codiag = codiag)
    else
        throw(ArgumentError("Could not establish diagonalizer method"))
    end
    return d
end

#######################################################################
# Codiagonalization
#######################################################################
struct Codiagonalizer{H}
    h::H
end

resolve_degeneracies!(ϵ, ψ, codiag::Missing) = (ϵ, ψ)
function resolve_degeneracies!(ϵ, ψ, codiag::Codiagonalizer{<:Hamiltonian})
    
end

#######################################################################
# Diagonalize methods
#   (All but LinearAlgebraPackage `@require` some package to be loaded)
#######################################################################
struct LinearAlgebraPackage{O} <: DiagonalizePackage
    options::O
end

LinearAlgebraPackage(; kw...) = LinearAlgebraPackage(values(kw))

# Registers method as available
diagonalizer(method::LinearAlgebraPackage, matrix; kw...) = Diagonalizer(method, matrix; kw...)

function diagonalize(d::Diagonalizer{<:LinearAlgebraPackage})
    ϵ, ψ = eigen!(d.matrix; sortby = λ -> abs(λ - d.origin), d.method.options...)
    resolve_degeneracies!(ϵ, ψ, d.codiag)
    return ϵ, ψ
end

# Fallback for unloaded packages

(m::DiagonalizePackage)(;kw...) =
    throw(ArgumentError("The package required for the requested diagonalize method $m is not loaded. Please do e.g. `using Arpack` to use the Arpack method. See `diagonalizer` for details."))

# Optionally loaded methods

struct ArpackPackage{O} <: DiagonalizePackage
    options::O
end

struct IterativeSolversPackage{O,L,E} <: DiagonalizePackage
    options::O
    point::Float64  # Shift point for shift and invert
    lmap::L         # LinearMap for shift and invert
    engine::E       # Optional support for lmap (e.g. Pardiso solver or factorization)
end

struct ArnoldiPackagePackage{O,L,E} <: DiagonalizePackage
    options::O
    point::Float64  # Shift point for shift and invert
    lmap::L         # LinearMap for shift and invert
    engine::E       # Optional support for lmap (e.g. Pardiso solver or factorization)
end
