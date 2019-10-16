#######################################################################
# Diagonalizer
#######################################################################
abstract type DiagonalizeMethod end

struct Diagonalizer{M<:DiagonalizeMethod,A<:AbstractArray,O<:NamedTuple,L,C,E}
    method::M
    matrix::A
    codiag::C       # Matrix to resolve degeneracies
end

Diagonalizer(method, matrix) = Diagonalizer(method, matrix, missing)

# This is in general type unstable, a function barrier when using it is needed
diagonalizer(h::Hamiltonian{<:Lattice,<:Any,<:Any,<:Matrix}; kw...) =
    diagonalizer(LinearAlgebraMethod(values(kw)), similarmatrix(h))

function diagonalizer(h::Hamiltonian{<:Lattice,<:Any,M,<:SparseMatrixCSC};
                      levels = missing, kw...) where {M}
    if size(h, 1) < 50 || levels === missing
        @warn "Requesting significant fraction of levels of sparse matrix. Converting to dense."
        matrix = Matrix(similarmatrix(h))
        wrappedmatrix = ishermitian(h) ? Hermitian(matrix) : matrix
        return diagonalizer(LinearAlgebraMethod(; kw...), wrappedmatrix)
    elseif M isa Number
        return diagonalizer(ArpackMethod(; kw...), similarmatrix(h))
    else
        return diagonalizer(ArpackMethod(; kw...), similarmatrix(h))
    end
    throw(ArgumentError("Could not establish diagonalizer method"))
end

resolve_degeneracies!(ϵ, ψ, codiag::Missing) = (ϵ, ψ)

#######################################################################
# Diagonalize methods
#   (All but LinearAlgebraMethod `@require` some package to be loaded)
#######################################################################
struct LinearAlgebraMethod{O} <: DiagonalizeMethod
    options::O
end

LinearAlgebraMethod(; kw...) = LinearAlgebraMethod(values(kw))

diagonalizer(method::LinearAlgebraMethod, matrix::Matrix) = Diagonalizer(method, matrix)

function diagonalize(d::Diagonalizer{<:LinearAlgebraMethod})
    ϵ, ψ = eigen!(d.matrix; d.options...)
    resolve_degeneracies!(ϵ, ψ, d.codiag)
    return ϵ, ψ
end

# Fallback for unloaded packages

(m::DiagonalizeMethod)(;kw...) =
    throw(ArgumentError("The package required for the requested diagonalize method $m is not loaded. Please do e.g. `using Arpack` to use the Arpack method. See `diagonalizer` for details."))

# Optionally loaded methods

struct ArpackMethod{O} <: DiagonalizeMethod
    levels::Int
    options::O
end

struct IterativeSolversMethod{O,L,E} <: DiagonalizeMethod
    levels::Int
    options::O
    point::Float64  # Shift point for shift and invert
    lmap::L         # LinearMap for shift and invert
    engine::E       # Optional support for lmap (e.g. Pardiso solver or factorization)
end

struct KrylovKitMethod{O,L,E} <: DiagonalizeMethod
    levels::Int
    options::O
    point::Float64  # Shift point for shift and invert
    lmap::L         # LinearMap for shift and invert
    engine::E       # Optional support for lmap (e.g. Pardiso solver or factorization)
end
