#######################################################################
# Diagonalize methods
#   (All but LinearAlgebraPackage `@require` some package to be loaded)
#######################################################################
abstract type AbstractDiagonalizePackage end

(package::Type{P})(;kw...) where {P<:AbstractDiagonalizePackage} = package(values(kw))

struct Diagonalizer{M<:AbstractDiagonalizePackage,A<:AbstractArray,C}
    method::M
    matrix::A
    levels::Int
    origin::Float64
    minprojection::Float64
    codiag::C         # Matrices to resolve degeneracies, or missing
    perm::Vector{Int} # Prealloc to sort eigenvalues/vectors
end

function Diagonalizer(method, matrix::AbstractMatrix{M}, opts) where {M}
    levels = opts.levels === missing ? size(matrix, 1) : opts.levels
    origin, minprojection, codiag = opts.origin, opts.minprojection, opts.codiag
    Diagonalizer(method, matrix, levels, origin, minprojection, codiag, Vector{Int}(undef, levels))
end

## LinearAlgebra ##
struct LinearAlgebraPackage{O} <: AbstractDiagonalizePackage
    opts::O
end

function diagonalize(d::Diagonalizer{<:LinearAlgebraPackage})
    ϵ, ψ = eigen!(d.matrix; d.method.opts...)
    ϵ´, ψ´ = selectstates(ϵ, ψ, d)
    return ϵ´, ψ´
end

function selectstates(ϵ, ψ, d)
    # Must implement selecting d.levels around origin
    # resize!(d.perm, length(ϵ))
    # sortperm!(d.perm, ϵ)
    ϵ´, ψ´ = view(ϵ, 1:d.levels), view(ψ, :, 1:d.levels)
    return ϵ´, ψ´
end

## Arpack ##
struct ArpackPackage{O} <: AbstractDiagonalizePackage
    opts::O
end

function diagonalize(d::Diagonalizer{<:ArpackPackage})
    ϵ, ψ = Main.Arpack.eigs(d.matrix; d.method.opts...)
    ϵ´ = real.(ϵ)
    length(ϵ´) == d.levels || throw(ArgumentError("Got a number of eigenvalues different from the required $(d.levels). Try specifying a different `levels`"))
    return ϵ´, ψ
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
# diagonalizer
#######################################################################
"""
    diagonalizer(h::Hamiltonian, mesh::Mesh; diagkw...)

Build an auxiliary `Diagonalizer` object that encondes options for the diagonalization of
Hamiltonian `h`

Diagonalization options `diagkw`

    diagkw          default
    -----------------------
    method          missing
    levels          missing
    origin          0.0
    minprojection   0.5

If `method` is `missing`, a method is automatically chosen between the following options

    LinearAlgebraPackage(; kw...)
    ArpackPackage(; kw...)          (must be `using Arpack`)

Keywords `kw` are forwarded to the appropriate `eigen` or equivalent of the chosen package.

# Example
```
julia> h = LatticePresets.honeycomb() |> unitcell(3) |> hamiltonian(hopping(-1, range = 1/√3));

julia> m = marchingmesh(h, npoints = 25);

julia> using Arpack; d = diagonalizer(h, m; levels = 2, method = ArpackPackage(maxiter = 300));

julia> bandstructure!(d, h, m)
Bandstructure: bands for a 2D hamiltonian
  Bands        : 3
  Element type : scalar (Complex{Float64})
  Mesh{2}: mesh of a 2-dimensional manifold
    Vertices   : 625
    Edges      : 3552
```

# See also
    marchingmesh, diagonalizer, bandstructure!
"""
function diagonalizer(h, mesh;
    method = missing, levels = missing, origin = 0.0, minprojection = 0.5)
    levels´ = levels === missing ? size(h, 1) : clamp(levels, 1, size(h, 1))
    opts = (levels = levels´, origin = origin, minprojection = minprojection,
            codiag = defaultcodiagonalizer(h, mesh))
    return diagonalizer(method, h, opts)
end

function diagonalizer(method::Missing, h, opts)
    if dense_method_heuristic(opts.levels, h)
        return diagonalizer(LinearAlgebraPackage(), h, opts)
    else
        # if we get here, opts.levels is an `Integer`
        return diagonalizer(ArpackPackage(; nev = opts.levels, sigma = 1im), h, opts)
    end
end

function diagonalizer(method::ArpackPackage, h::Hamiltonian{<:Lattice,<:Any,<:Number,<:SparseMatrixCSC}, opts)
    matrix = similarmatrix(h)
    matrix´ = ishermitian(h) ? Hermitian(matrix) : matrix
    method´ = ArpackPackage((method.opts..., nev = opts.levels, sigma = opts.origin + 1im))
    return Diagonalizer(method´, matrix´, opts)
end

function diagonalizer(method::LinearAlgebraPackage, h, opts)
    matrix = Matrix(similarmatrix(h))
    matrix´ = ishermitian(h) ? Hermitian(matrix) : matrix
    Diagonalizer(method, matrix´, opts)
end

# fallback
diagonalizer(method, h, opts) =
    throw(ArgumentError("Could not establish diagonalizer method, or specified options are incompatible"))

dense_method_heuristic(levels::Missing, h) = true
dense_method_heuristic(levels::Integer, h) =
    issparse(h) && (size(h, 1) < 50 || levels / size(h, 1) > 0.1)

#######################################################################
# resolve_degeneracies
#######################################################################
# Tries to make states continuous at crossings. Here, ϵ needs to be sorted
function resolve_degeneracies!(ϵ, ψ, d::Diagonalizer, ϕs)
    issorted(ϵ) || sorteigs!(d.perm, ϵ, ψ)
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

# Could perhaps be better/faster using a generalized CoSort
function sorteigs!(perm, ϵ::Vector{<:Real}, ψ::Matrix)
    resize!(perm, length(ϵ))
    p = sortperm!(perm, ϵ)
    # permute!(ϵ, p)
    sort!(ϵ)
    Base.permutecols!!(ψ, p)
    return ϵ, ψ
end

#######################################################################
# Codiagonalizers
#######################################################################
defaultcodiagonalizer(h, mesh) = VelocityCodiagonalizer(h, meshdelta(mesh))

meshdelta(mesh::Mesh{<:Any,T}) where {T} = T(0.1) * first(minmax_edge_length(mesh))

## VelocityCodiagonalizer
## Uses velocity operators along different directions. If not enough, use finite differences
struct VelocityCodiagonalizer{S,T,H<:Hamiltonian}
    h::H
    directions::Vector{S}
    degtol::T
    delta::T                        # Finite differences
    rangesA::Vector{UnitRange{Int}} # Prealloc buffer for degeneray ranges
    rangesB::Vector{UnitRange{Int}} # Prealloc buffer for degeneray ranges
end

function VelocityCodiagonalizer(h::Hamiltonian{<:Any,L}, delta;
                                direlements = -0:1, onlypositive = true, kw...) where {L}
    directions = vec(SVector{L,Int}.(Iterators.product(ntuple(_ -> direlements, Val(L))...)))
    onlypositive && filter!(ispositive, directions)
    unique!(normalize, directions)
    sort!(directions, by = norm, rev = false)
    degtol = sqrt(eps(realtype(h)))
    delta´ = delta === missing ? degtol : delta
    VelocityCodiagonalizer(h, directions, degtol, delta´, UnitRange{Int}[], UnitRange{Int}[])
end

num_codiag_matrices(d) = 2 * length(d.codiag.directions)
function codiag_matrix(n, d, ϕs)
    ndirs = length(d.codiag.directions)
    if n <= ndirs
        bloch!(d.matrix, d.codiag.h, ϕs, dn -> im * d.codiag.directions[n]' * dn)
    else # resort to finite differences
        bloch!(d.matrix, d.codiag.h, ϕs + d.codiag.delta * d.codiag.directions[n - ndirs])
    end
    return d.matrix
end