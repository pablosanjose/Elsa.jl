#######################################################################
# Brillouin
#######################################################################
"""
    Brillouin(lat::Lattice; uniform::Bool = false, partitions = 5)

Discretizes the Brillouin zone of Lattice `lat` into hypertriangular finite
elements, using a certain number of `partitions` per Bravais axis (accepts
an integer or a tuple of integers, one per axis). Keyword `uniform` specifies
the type of mesh, either "uniform" (as close to equilateral as possible) or
"simple" (cartesian partition)

# Examples
```jldoctest
julia> Brillouin(Lattice(:honeycomb), uniform = true, partitions = 200)
Brillouin{Float64,2} : discretization of 2-dimensional Brillouin zone
    Mesh type  : uniform
    Vertices   : 40000
    Partitions : (200, 200)
    3-elements : 80000
```
"""
struct Brillouin{T,L,LL}
    lattice::Lattice{T,L,L,LL}
    uniform::Bool
    partitions::NTuple{L,Int}
end

function Brillouin(lat::Lattice{T,E,L}; uniform::Bool = false, partitions = 5) where {T,E,L}
    partitions_tuple = tontuple(Val(L), partitions)
    if uniform
        lattice = uniform_discretization(lat, partitions_tuple)
    else
        lattice = simple_discretization(lat, partitions_tuple)
    end
    return Brillouin(lattice, uniform, partitions_tuple)
end
Brillouin(sys::System; kw...) = Brillouin(sys.lattice; kw...)

Base.show(io::IO, m::Brillouin{T,L,N}) where {T,L,N} =
    print(io, "Brillouin{$T,$L} : discretization of $L-dimensional Brillouin zone
    Mesh type  : $(m.uniform ? "uniform" : "simple")
    Vertices   : $(nsites(m.lattice))
    Partitions : $(m.partitions)")

# phi-space sampling z, k-space G'z. M = diagonal(partitions)
# M G' z =  Tr * n, where n are SVector{L,Int}, and Tr is a hypertriangular lattice
# For some integer S = (n1,n2...), (z1, z2, z3) = I (corners of BZ).
# Hence S = round.(Tr^{-1} G' M) = supercell. Bravais are z_i for n = I, so simply S^{-1}
# Links should be fixed at the Tr level, then transform so that D * Tr = S^{-1}, and do Supercell(S)
function uniform_discretization(lat::Lattice{T,E,L}, partitions_tuple::NTuple{L,Int}) where {T,E,L}
    M = diagsmatrix(partitions_tuple)
    A = qr(bravaismatrix(lat)).R
    Gt = qr(transpose(inv(A))).R
    Gt = Gt / Gt[1,1]
    Tr = hypertriangularbravais(lat)
    S = round.(Int, inv(Tr) * Gt * M)
    iS = inv(S)
    D = iS * inv(Tr)
    meshlat = Lattice(Sublat(zero(SVector{L,T})), Bravais(Tr), LinkRule(1))
    meshlat = transform!(meshlat, r -> D * r)
    meshlat = lattice!(meshlat, Supercell(S))
    # methlat = transform!(meshlat, r -> Gt * r)  # to go back to k space
    return meshlat
end

# Transformation D takes hypertriangularbravais to minisquare (phi-space delta): D * Tr = M^{-1}
# However, Tr has to be chosen to match bravais angles of G' as much as possible: Trsigned
# Build+link Trsigned, transform, Supercell(M)
function simple_discretization(lat::Lattice{T,E,L}, partitions_tuple::NTuple{L,Int}) where {T,E,L}
    M = diagsmatrix(partitions_tuple)
    A = qr(bravaismatrix(lat)).R
    Gt = qr(transpose(inv(A))).R
    Gt = Gt / Gt[1,1]
    Tr = hypertriangularbravais(lat)
    Trsigned = zero(MMatrix{L,L,T})
    for i in 1:L
        Trsigned[1:L, i] .= Tr[1:L, i] * sign_positivezero(Gt[1,i])
    end
    D = inv(Trsigned * M)
    meshlat = Lattice(Sublat(zero(SVector{L,T})), Bravais(Trsigned), LinkRule(1))
    meshlat = transform!(meshlat, r -> D * r)
    meshlat = lattice!(meshlat, Supercell(M))
    return meshlat
end

# Completes the L2 vectors into an L-dimensional basis s with new vectors at 60 degree angles
hypertriangularbravais(lat::Lattice{T,E,L}) where {T,E,L} = hypertriangularbravais(SMatrix{L,1,T}(I))
function hypertriangularbravais(s::SMatrix{L,L2,T}) where {L,L2,T}
    v1 = s[:,L2]
    factor = T(1/(L2+1))
    v2 = modifyat(v1, L2, v1[L2]*factor)
    v2 = modifyat(v2, L2 + 1, v1[L2]*sqrt(1 - factor^2))
    return hypertriangularbravais(hcat(s, v2))
end
hypertriangularbravais(s::SMatrix{L,L}) where L = s

#######################################################################
# BandSampling
#######################################################################

struct BandSampling{T<:Real,L}
    energies::Matrix{T}
    nenergies::Int
    states::Array{Complex{T},3}
    statelength::Int
    points::Vector{SVector{L,T}}
    npoints::Int
    bufferstate::Vector{Complex{T}}
end

function BandSampling(hfunc::Function, dimh, lat::Lattice{T,E,L}, vfunc; levels = missing, degtol = sqrt(eps()), randomshift = missing, kw...) where {T,E,L}
    shift = ismissing(randomshift) ? zero(SVector{E,T}) : randomshift * rand(SVector{E,T})
    points = lat.sublats[1].sites
    npoints = length(points)

    nenergies = ismissing(levels) ? dimh : 2
    statelength = dimh

    preallocH = Matrix{Complex{T}}(undef, (dimh, dimh))

    ordering = zeros(Int, nenergies)
    energies = Matrix{T}(undef, (nenergies, npoints))
    states = Array{Complex{T},3}(undef, (statelength, nenergies, npoints))

    @showprogress "Diagonalising: " for n in 1:npoints
        (energies_n, states_n) = spectrum(hfunc(points[n] + shift), preallocH; levels = nenergies, kw...)
        # sort_spectrum!(energies_nk, states_nk, ordering)
        resolve_degeneracies!(energies_n, states_n, vfunc, points[n], degtol)
        copyslice!(energies,    CartesianIndices((1:nenergies, n:n)),
                   energies_n, CartesianIndices(1:nenergies))
        copyslice!(states,      CartesianIndices((1:statelength, 1:nenergies, n:n)),
                   states_n,   CartesianIndices((1:statelength, 1:nenergies)))
    end

    bufferstate = zeros(Complex{T}, statelength)

    return BandSampling(energies, nenergies, states, statelength, points, npoints, bufferstate)
end

function spectrum(h::SparseMatrixCSC, preallocH; levels = 2, method = missing, kw...)
    if method === :exact || ismissing(method) && (size(h, 1) < 129 || levels/size(h,1) > 0.2)
        s = spectrum_dense(h, preallocH; levels = levels, kw...)
    elseif method === :arnoldi
        s = spectrum_arpack(h; levels = levels, kw...)
    else
        throw(ArgumentError("Unknown method. Choose between :arnoldi or :exact"))
    end
    return s
end

function spectrum_dense(h::SparseMatrixCSC, preallocH; levels = 2, kw...)
    preallocH .= h
    dimh = size(h, 1)
    range = ((dimh - levels)÷2 + 1):((dimh + levels)÷2)
    ee = eigen!(Hermitian(preallocH), range, kw...)
    energies, states = ee.values, ee.vectors
    return (energies, states)
end

function spectrum_arpack(h::SparseMatrixCSC; levels = 2, sigma = 1.0im, kw...)
    (energies, states, _) = eigs(h; sigma = sigma, nev = levels, kw...)
    return (real.(energies), states)
end

# function spectrum_arnoldimethods(hsparse::SparseMatrixCSC{T}, alg; nenergies = 2, kw...) where {T}
#     h = Hermitian(hsparse)
#     F = cholesky(H, check = false)
#     L = ldlt!(F, H, shift = eps())
#     map = LinearMap{T}(x -> F \ x, size(hsparse, 1))
#     schur = partialschur(map, nev = nenergies, tol=1e-6, restarts = 100, which = LM())
#     energies, states = partialeigen(schur)
#     energies = 1 ./ schur.eigenvalues
#     return (energies, states)
# end

# function sort_spectrum!(energies, states, ordering)
#     if !issorted(energies)
#         sortperm!(ordering, energies)
#         energies .= energies[ordering]
#         states .= states[:, ordering]
#     end
#     return (energies, states)
# end

function hasdegeneracies(energies, degtol)
    has = false
    for i in eachindex(energies), j in (i+1):length(energies)
        if abs(energies[i] - energies[j]) < degtol
            has = true
            break
        end
    end
    return has
end

function degeneracies(energies, degtol)
    if hasdegeneracies(energies, degtol)
        deglist = Vector{Int}[]
        isclassified = BitArray(false for _ in eachindex(energies))
        for i in eachindex(energies)
            isclassified[i] && continue
            degeneracyfound = false
            for j in (i + 1):length(energies)
                if !isclassified[j] && abs(energies[i] - energies[j]) < degtol
                    !degeneracyfound && push!(deglist, [i])
                    degeneracyfound = true
                    push!(deglist[end], j)
                    isclassified[j] = true
                end
            end
        end
        return deglist
    else
        return nothing
    end
end

resolve_degeneracies!(energies, states, vfunc::Missing, kn, degtol) = nothing
function resolve_degeneracies!(energies, states, vfunc::Function, kn::SVector{L}, degtol) where {L}
    degsubspaces = degeneracies(energies, degtol)
    if !(degsubspaces === nothing)
        for subspaceinds in degsubspaces
            for axis = 1:L
                v = vfunc(kn, axis)  # Need to do it in-place for each subspace
                subspace = view(states, :, subspaceinds)
                vsubspace = subspace' * v * subspace
                veigen = eigen!(vsubspace)
                subspace .= subspace * veigen.vectors
                success = !hasdegeneracies(veigen.values, degtol)
                success && break
            end
        end
    end
    return nothing
end

#######################################################################
# Bandstructure
#######################################################################

struct Bandstructure{T,N,L,NL}  # E = N = L + 1 (nodes are [Blochphases..., energy])
    bands::Mesh{T,N,L,N,NL}
    states::Matrix{Complex{T}}
    nenergies::Int
    npoints::Int
end

Base.show(io::IO, bs::Bandstructure{T,N,L}) where {T,N,L} =
    print(io, "Bandstructure{$T,$N,$L} of type $T for $L-dimensional lattice
    Number of k-points    : $(bs.npoints)
    Number of eigenvalues : $(bs.nenergies)
    Size of state vectors : $(size(bs.states, 1))")

Bandstructure(sys::System; uniform = false, partitions = 5, kw...) =
    Bandstructure(sys, Brillouin(sys.lattice; uniform = uniform, partitions = partitions); kw...)
Bandstructure(sys::System{T,E,L}, brillouin::Brillouin{T,L}; kw...) where {T,E,L} =
    Bandstructure(kn -> hamiltonian!(sys, kn = kn), hamiltoniandim(sys), brillouin.lattice;
                  velocity = (kn, axis) -> velocity!(sys, kn = kn, axis = axis), kw...)
Bandstructure(hfunc::Function, lat::Lattice{T,E}; kw...) where {T,E} =
    Bandstructure(hfunc, hamiltoniandim(hfunc(zero(SVector{E,T}))), lat; kw...)
Bandstructure(hfunc::Function, ranges::AbstractVector...; kw...) =
    Bandstructure(hfunc, productlattice(ranges...); kw...)


function Bandstructure(hfunc::Function, hdim, lat::Lattice; velocity = missing, linkthreshold = 0.5, kw...)
    isunlinked(lat) && throw(ErrorException("The band sampling lattice is not linked"))
    bandsampling = BandSampling(hfunc, hdim, lat, velocity; kw...)
    bandslat = bandslattice(lat, bandsampling, linkthreshold)
    states = reshape(bandsampling.states, bandsampling.statelength, :)
    bandsmesh = Mesh(bandslat)
    return Bandstructure(bandsmesh, states, bandsampling.nenergies, bandsampling.npoints)
end

function bandslattice(lat::Lattice{T,E,L}, bandsampling, linkthreshold) where {T,E,L}
    bands = Lattice(Sublat{T,E+1}(), Bravais(SMatrix{E+1,L,T}(I)))
    addnodes!(bands, bandsampling)
    for samplingilink in allilinks(lat)
        addilink!(bands.links, samplingilink, bandsampling, linkthreshold, bands)
    end
    return bands
end

function addnodes!(bands, bandsampling)
    bandnodes = bands.sublats[1].sites
    for nk in 1:bandsampling.npoints, ne in 1:bandsampling.nenergies
        push!(bandnodes, vcat(bandsampling.points[nk], bandsampling.energies[ne, nk]))
    end
    return bands
end

function addilink!(bandlinks::Links, samplingilink::Ilink, sp::BandSampling, linkthreshold, bands)
    bandnodes = bands.sublats[1,1].sites
    dist = bravaismatrix(bands) * samplingilink.ndist
    linearindices = LinearIndices(sp.energies)
    state = sp.bufferstate
    states = sp.states

    slinkbuilder = SparseMatrixBuilder(bands, 1, 1)
    neighiter = NeighborIterator(samplingilink, 1, (1,1))

    @showprogress "Linking bands: " for nk_src in 1:sp.npoints, ne_src in 1:sp.nenergies
        n_src = linearindices[ne_src, nk_src]
        r1 = bandnodes[n_src]
        copyslice!(state,  CartesianIndices(1:sp.statelength),
                   states, CartesianIndices((1:sp.statelength, ne_src:ne_src, nk_src:nk_src)))
        @inbounds for nk_target in neighbors!(neighiter, nk_src)
            ne_target = findmostparallel(state, states, nk_target, linkthreshold)
            if !iszero(ne_target)
                n_target = linearindices[ne_target, nk_target]
                r2 = bandnodes[n_target] + dist
                pushtocolumn!(slinkbuilder, n_target, _rdr(r1, r2))
            end
        end
        finalisecolumn!(slinkbuilder)
    end
    push!(bandlinks, Ilink(samplingilink.ndist, fill(Slink(sparse(slinkbuilder)), 1, 1)))
    return bandlinks
end

function findmostparallel(state::Vector{Complex{T}}, states, ktarget, linkthreshold) where {T}
    maxproj = T(linkthreshold)
    dotprods = abs.(state' * view(states, :, :, ktarget))
    proj, ne = findmax(dotprods)
    return proj > maxproj ? ne[2] : 0
end
