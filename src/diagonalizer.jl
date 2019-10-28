#######################################################################
# Diagonalizer
#######################################################################
abstract type AbstractCodiagonalizer end
abstract type AbstractDiagonalizePackage end

struct Diagonalizer{M<:AbstractDiagonalizePackage,A<:AbstractArray,C<:Union{Missing,AbstractCodiagonalizer}}
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
# Diagonalize methods
#   (All but LinearAlgebraPackage `@require` some package to be loaded)
#######################################################################
struct LinearAlgebraPackage{O} <: AbstractDiagonalizePackage
    options::O
end

LinearAlgebraPackage(; kw...) = LinearAlgebraPackage(values(kw))

# Registers method as available
diagonalizer(method::LinearAlgebraPackage, matrix; kw...) = Diagonalizer(method, matrix; kw...)

function diagonalize(d::Diagonalizer{<:LinearAlgebraPackage})
    ϵ, ψ = eigen!(d.matrix; sortby = λ -> abs(λ - d.origin), d.method.options...)
    # ϵ´, ψ´ = view(ϵ, 1:d.levels), view(ψ, :, 1:d.levels)
    # return ϵ´, ψ´
end

# Fallback for unloaded packages

(m::AbstractDiagonalizePackage)(;kw...) =
    throw(ArgumentError("The package required for the requested diagonalize method $m is not loaded. Please do e.g. `using Arpack` to use the Arpack method. See `diagonalizer` for details."))

# Optionally loaded methods

struct ArpackPackage{O} <: AbstractDiagonalizePackage
    options::O
end

struct IterativeSolversPackage{O,L,E} <: AbstractDiagonalizePackage
    options::O
    point::Float64  # Shift point for shift and invert
    lmap::L         # LinearMap for shift and invert
    engine::E       # Optional support for lmap (e.g. Pardiso solver or factorization)
end

struct ArnoldiPackagePackage{O,L,E} <: AbstractDiagonalizePackage
    options::O
    point::Float64  # Shift point for shift and invert
    lmap::L         # LinearMap for shift and invert
    engine::E       # Optional support for lmap (e.g. Pardiso solver or factorization)
end


#######################################################################
# Codiagonalization
#######################################################################
struct VelocityCodiagonalizer{M,H<:Hamiltonian{<:Any,<:Any,M}} <: AbstractCodiagonalizer
    h::H
    degranges::Vector{UnitRange{Int}}
    success::Vector{Bool}
end
VelocityCodiagonalizer(h::Hamiltonian{<:Any,<:Any,M}) where {M} =
    VelocityCodiagonalizer(h, Int[], Bool[])

# ϵ is assumed sorted
resolve_degeneracies!(ϵ, ψ, d::Diagonalizer{<:Any,<:Any,Missing}, ϕs) = (ϵ, ψ)

function resolve_degeneracies!(ϵ, ψ, d::Diagonalizer{<:Any,<:Any,<:AbstractCodiagonalizer}, ϕs)
    issorted(ϵ) || throw(ArgumentError("Unsorted eigenvalues"))
    if hasdegeneracies(ϵ)
        finddegeneracies!(d.codiag.degranges, ϵ)
    else
        return ϵ, ψ
    end
    success = d.codiag.success
    resize!(success, length(d.codiag.degranges))
    fill!(success, false)
    for v in codiag_matrices(d.codiag, ϕs)
        all(success) && break
        for (i,r) in enumerate(d.codiag.degranges)
            success[i] || (success[i] = codiagonalize!(ϵ, ψ, v, r))
        end
    end
    return ϵ, ψ
end

function hasdegeneracies(sorted_ϵ)
    for i in 2:length(sorted_ϵ)
        sorted_ϵ[i] ≈ sorted_ϵ[i-1] && return true
    end
    return false
end

finddegeneracies!(degranges, sorted_ϵ) = approxruns!(degranges, sorted_ϵ)

codiag_matrices(d::Diagonalizer, ϕs::SVector{L}) where {L} =
    (bloch!(d.matrix, d.codiag.h, ϕs; axis = i) for i in 1:L)

function codiagonalize!(ϵ, ψ, v, r)
    subspace = view(ψ, :, r)
    vsubspace = subspace' * v * subspace
    veigen = eigen!(vsubspace)
    subspace .= subspace * veigen.vectors
    success = !hasdegeneracies(veigen.values)
    return success
end

# function resolve_degeneracies!(energies, states, vfunc::Function, kn::SVector{L}, degtol) where {L}
#     degsubspaces = degeneracies(energies, degtol)
#     if !(degsubspaces === nothing)
#         for subspaceinds in degsubspaces
#             for axis = 1:L
#                 v = vfunc(kn, axis)  # Need to do it in-place for each subspace
#                 subspace = view(states, :, subspaceinds)
#                 vsubspace = subspace' * v * subspace
#                 veigen = eigen!(vsubspace)
#                 subspace .= subspace * veigen.vectors
#                 success = !hasdegeneracies(veigen.values, degtol)
#                 success && break
#             end
#         end
#     end
#     return nothing
# end

# function degeneracies(energies, degtol)
#     if hasdegeneracies(energies, degtol)
#         deglist = Vector{Int}[]
#         isclassified = BitArray(false for _ in eachindex(energies))
#         for i in eachindex(energies)
#             isclassified[i] && continue
#             degeneracyfound = false
#             for j in (i + 1):length(energies)
#                 if !isclassified[j] && abs(energies[i] - energies[j]) < degtol
#                     !degeneracyfound && push!(deglist, [i])
#                     degeneracyfound = true
#                     push!(deglist[end], j)
#                     isclassified[j] = true
#                 end
#             end
#         end
#         return deglist
#     else
#         return nothing
#     end
# end