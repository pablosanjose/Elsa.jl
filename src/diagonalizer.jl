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
                      levels, origin, minprojection, codiag) where {M}
    _levels = levels === missing ? size(matrix, 1) : levels
    Diagonalizer(method, matrix, _levels, origin, minprojection, codiag)
end

# This is in general type unstable. A function barrier when using it is needed
diagonalizer(h::Hamiltonian{<:Lattice,<:Any,<:Any,<:Matrix}, mesh = missing; kw...) =
    diagonalizer(LinearAlgebraPackage(values(kw)), similarmatrix(h); kw...)

function diagonalizer(h::Hamiltonian{<:Lattice,<:Any,M,<:SparseMatrixCSC}, mesh = missing;
                      levels = missing, origin = 0.0,
                      codiag = defaultcodiagonalizer(h, mesh), minprojection = 0.7,
                      methodkw...) where {M}
    diagkw = (levels = levels, origin = origin, codiag = codiag, minprojection = minprojection)
    if size(h, 1) < 50 || levels === missing || levels / size(h, 1) > 0.1
        # @warn "Requesting significant number of sparse matrix eigenvalues. Converting to dense."
        matrix = Matrix(similarmatrix(h))
        _matrix = ishermitian(h) ? Hermitian(matrix) : matrix
        d = Diagonalizer(LinearAlgebraPackage(; methodkw...), _matrix; diagkw...)
    elseif M <: Number
        matrix = similarmatrix(h)
        _matrix = ishermitian(h) ? Hermitian(matrix) : matrix
        d = Diagonalizer(ArpackPackage(; nev = levels, methodkw...), _matrix; diagkw...)
    elseif M <: SMatrix
        matrix = similarmatrix(h)
        _matrix = ishermitian(h) ? Hermitian(matrix) : matrix
        d = Diagonalizer(ArnoldiPackagePackage(; methodkw...), _matrix; diagkw...)
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

function diagonalize(d::Diagonalizer{<:LinearAlgebraPackage})
    ϵ, ψ = eigen!(d.matrix; d.method.options...)
    ϵ´, ψ´ = view(ϵ, 1:d.levels), view(ψ, :, 1:d.levels)
    return ϵ´, ψ´
end

# Fallback for unloaded packages

(m::AbstractDiagonalizePackage)(;kw...) =
    throw(ArgumentError("The package required for the requested diagonalize method $m is not loaded. Please do e.g. `using Arpack` to use the Arpack method. See `diagonalizer` for details."))

# Optionally loaded methods

## Arpack ##
struct ArpackPackage{O} <: AbstractDiagonalizePackage
    options::O
    perm::Vector{Int}
end

function diagonalize(d::Diagonalizer{<:ArpackPackage})
    ϵ, ψ = Arpack.eigs(d.matrix; d.method.options...)
    ϵ´ = real.(ϵ)
    ϵ´, ψ´ = sorteigs!(d.method.perm, ϵ´, ψ)
    return ϵ´, ψ´
end

# struct IterativeSolversPackage{O,L,E} <: AbstractDiagonalizePackage
#     options::O
#     point::Float64  # Shift point for shift and invert
#     lmap::L         # LinearMap for shift and invert
#     engine::E       # Optional support for lmap (e.g. Pardiso solver or factorization)
# end

# struct ArnoldiPackagePackage{O,L,E} <: AbstractDiagonalizePackage
#     options::O
#     point::Float64  # Shift point for shift and invert
#     lmap::L         # LinearMap for shift and invert
#     engine::E       # Optional support for lmap (e.g. Pardiso solver or factorization)
# end

#######################################################################
# resolve_degeneracies
#######################################################################
# Tries to make states continuous at crossings. Here, ϵ needs to be sorted
resolve_degeneracies!(ϵ, ψ, d::Diagonalizer{<:Any,<:Any,Missing}, ϕs) = (ϵ, ψ)

function resolve_degeneracies!(ϵ, ψ, d::Diagonalizer{<:Any,<:Any,<:AbstractCodiagonalizer}, ϕs)
    issorted(ϵ) || throw(ArgumentError("Unsorted eigenvalues"))
    hasapproxruns(ϵ, d.codiag.degtol) || return ϵ, ψ
    ranges, ranges´ = d.codiag.rangesA, d.codiag.rangesB
    resize!(ranges, 0)
    pushapproxruns!(ranges, ϵ, 0, d.codiag.degtol) # 0 is an offset
    for n in 1:num_codiag_matrices(d)
        v = codiag_matrix(n, d, ϕs)
        resize!(ranges´, 0)
        for (i, r) in enumerate(ranges)
            subspace = view(ψ, :, r)
            vsubspace = subspace' * v * subspace
            veigen = eigen!(Hermitian(vsubspace))
            if hasapproxruns(veigen.values, d.codiag.degtol)
                roffset = minimum(r) - 1 # Range offset within the ϵ vector
                pushapproxruns!(ranges´, veigen.values, roffset, d.codiag.degtol)
            end
            subspace .= subspace * veigen.vectors
        end
        ranges, ranges´ = ranges´, ranges
        isempty(ranges) && break
    end
    return ψ
end

function sorteigs!(perm, ϵ::Vector{<:Real}, ψ::Matrix)
    p = sortperm!(perm, ϵ)
    # permute!(ϵ, p)
    sort!(ϵ)
    Base.permutecols!!(ψ, p)
    return ϵ, ψ
end

#######################################################################
# Codiagonalizers
#######################################################################
defaultcodiagonalizer(h, mesh) = VelocityCodiagonalizer(h, meshshift(mesh))

meshshift(::Missing) = missing
meshshift(mesh::Mesh{<:Any,T}) where {T} = T(0.1) * first(minmax_edge_length(mesh))

## VelocityCodiagonalizer
## Uses velocity operators along different directions. If not enough, use finite differences
struct VelocityCodiagonalizer{S,T,H<:Hamiltonian} <: AbstractCodiagonalizer
    h::H
    directions::Vector{S}
    degtol::T
    shift::T
    rangesA::Vector{UnitRange{Int}} # Prealloc buffer for degeneray ranges
    rangesB::Vector{UnitRange{Int}} # Prealloc buffer for degeneray ranges
end

function VelocityCodiagonalizer(h::Hamiltonian{<:Any,L}, shift;
                                direlements = -0:1, onlypositive = true, kw...) where {L}
    directions = vec(SVector{L,Int}.(Iterators.product(ntuple(_ -> direlements, Val(L))...)))
    onlypositive && filter!(ispositive, directions)
    unique!(normalize, directions)
    sort!(directions, by = norm, rev = false)
    degtol = sqrt(eps(realtype(h)))
    shift´ = shift === missing ? degtol : shift
    VelocityCodiagonalizer(h, directions, degtol, shift´, UnitRange{Int}[], UnitRange{Int}[])
end

num_codiag_matrices(d::Diagonalizer{<:Any,<:Any,<:VelocityCodiagonalizer}) =
    2 * length(d.codiag.directions)
function codiag_matrix(n, d::Diagonalizer{<:Any,<:Any,<:VelocityCodiagonalizer}, ϕs)
    ndirs = length(d.codiag.directions)
    if n <= ndirs
        bloch!(d.matrix, d.codiag.h, ϕs, dn -> im * d.codiag.directions[n]' * dn)
    else # resort to finite differences
        bloch!(d.matrix, d.codiag.h, ϕs + d.codiag.shift * d.codiag.directions[n - ndirs])
    end
    return d.matrix
end