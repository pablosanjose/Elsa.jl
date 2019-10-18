#######################################################################
# Spectrum and Bandstructure
#######################################################################
struct Spectrum{M,T}
    states::Matrix{M}
    energies::Vector{T}
end

struct Band{M,T,D,MD<:Mesh{D}}
    states::Matrix{Union{M,Missing}}
    energies::Vector{Union{T,Missing}}
    mesh::MD
end

function Band{M,T}(mesh::Mesh, dimh::Int) where {M,T}
    nk = nvertices(mesh)
    states = Union{M,Missing}[missing for _ in 1:dimh, _ in 1:nk]
    energies = Union{T,Missing}[missing for _ in 1:nk]
    return Band(states, energies, mesh)
end

struct Bandstructure{M,T,D,MD<:Mesh{D}}   # D is dimension of parameter space
    bands::Vector{Band{M,T,D,MD}}
    mesh::MD
end

function Base.show(io::IO, b::Bandstructure{M,T,D}) where {M,T,D}
    ioindent = IOContext(io, :indent => string("  "))
    print(io,
"Bandstructure: bands for a $(D)D hamiltonian
  Bands        : $(length(b.bands))
  Element type : $(displayelements(M))")
    print(ioindent, "\n", b.mesh)
end

#######################################################################
# bandstructure
#######################################################################

bandstructure(h::Hamiltonian{<:Any, L}, mesh = marchingmesh(ntuple(_ -> 10, Val(L))); kw...) where {L} =
    bandstructure!(diagonalizer(h; kw...), h, mesh) # barrier for type-unstable diagonalizer

function bandstructure!(d::Diagonalizer, h::Hamiltonian{<:Lattice,<:Any,M}, mesh::MD) where {M,D,MD<:Mesh{D}}
    T = eltype(M)
    nϵ = d.levels
    dimh = size(h, 1)
    nk = nvertices(mesh)
    ϵks = Matrix{T}(undef, nϵ, nk)
    ψks = Array{M,3}(undef, dimh, nϵ, nk)
    p = Progress(nk, "Step 1/2 - Diagonalising: ")
    for (n, ϕs) in enumerate(vertices(mesh))
        bloch!(d.matrix, h, ϕs)
        (ϵk, ψk) = diagonalize(d)
        copyslice!(ϵks, CartesianIndices((1:nϵ, n:n)),
                   ϵk,  CartesianIndices((1:nϵ,)))
        copyslice!(ψks, CartesianIndices((1:dimh, 1:nϵ, n:n)),
                   ψk,  CartesianIndices((1:dimh, 1:nϵ)))
        ProgressMeter.next!(p; showvalues = ())
    end
    bands = Band{M,T,D,MD}[Band{M,T}(mesh, dimh) for _ in 1:nϵ]
    # seed bands
    @inbounds for (nb, band) in enumerate(bands), i in 1:dimh
        band.states[i, 1] = ψks[i, nb, 1]
        band.energies[1] = ϵks[nb, 1]
    end
    p = Progress(nk, "Step 2/2 - Connecting bands: ")
    for src in 1:nk
        for edge in edges(mesh, src)
            dst = edgedest(mesh, edge)
            for band in bands
                proj, bandidx = findmostparallel(ψks, dst, band, src)
                if proj > d.minprojection
                    copyslice!(band.states, CartesianIndices((1:dimh, dst:dst)),
                            ψks, CartesianIndices((1:dimh, bandidx:bandidx, dst:dst)))
                    copyslice!(band.energies, CartesianIndices((dst:dst)),
                            ϵks  , CartesianIndices((bandidx:bandidx, dst:dst)))
                end
            end
        end
        ProgressMeter.next!(p; showvalues = ())
    end
    return Bandstructure(bands, mesh)
end

function findmostparallel(ψks::Array{M,3}, dst, band, src) where {M}
    T = real(eltype(M))
    dimh, nϵ, nk = size(ψks)
    maxproj = zero(T)
    idx = 0
    any(i -> band.states[i, src] === missing, 1:dimh) && return maxproj, idx
    @inbounds for j in 1:nϵ
        proj = zero(M)
        @simd for i in 1:dimh
            proj += ψks[i, j, dst]' * band.states[i, src]
        end
        absproj = T(abs(tr(proj)))
        if maxproj < absproj
            idx = j
            maxproj = absproj
        end
    end
    return maxproj, idx
end
#######################################################################
# Old
#######################################################################

# function spectrum(hfunc::Function, argmesh::Mesh;
#                   levels::Union{Int,Missing} = missing, degtol = sqrt(eps()),
#                   randomshift = missing, kw...)
#     hsample = hfunc(first(argmesh.verts)...)
#     Tv = eltype(hsample)
#     dimh = ψlen = size(hsample, 1)
#     nϵ = levels === missing ? dimh : levels
#     nk = nvertices(argmesh)
#     eigvals = Matrix{Tv}(undef, (nϵ, nk))
#     eigvecs = Array{Tv,3}(undef, (ψlen, nϵ, nk))
#     hwork = Matrix{Tv}(undef, (dimh, dimh))
#     @showprogress "Diagonalising: " for (n,k) in enumerate(argmesh.verts)
#         (ϵk, ψk) = _spectrum(hfunc(k...), hwork; levels = nϵ, kw...)
#         # sort_spectrum!(energies_nk, states_nk, ordering)
#         # resolve_degeneracies!(ϵk, ψk, vfunc, k, degtol)
#         copyslice!(eigvals, CartesianIndices((1:nϵ, n:n)),
#                    ϵk,      CartesianIndices(1:nϵ))
#         copyslice!(eigvecs, CartesianIndices((1:ψlen, 1:nϵ, n:n)),
#                    ψk,      CartesianIndices((1:ψlen, 1:nϵ)))
#     end
# end

# function _spectrum(h, hwork; levels = 2, method = missing, kw...)
#     if method === :exact || method === missing && (size(h, 1) < 129 || levels/size(h,1) > 0.2)
#         s = spectrum_direct(h, hwork; levels = levels, kw...)
#     elseif method === :arnoldi
#         s = spectrum_arpack(h; levels = levels, kw...)
#     else
#         throw(ArgumentError("Unknown method. Choose between :arnoldi or :exact"))
#     end
#     return s
# end

# function spectrum_direct(h, hwork; levels = 2, kw...)
#     hwork .= h
#     dimh = size(h, 1)
#     range = ((dimh - levels)÷2 + 1):((dimh + levels)÷2)
#     ee = eigen!(Hermitian(hwork), range, kw...)
#     energies, states = ee.values, ee.vectors
#     return (energies, states)
# end

# function spectrum_arpack(h; levels = 2, sigma = 1.0im, kw...)
#     (energies, states, _) = eigs(h; sigma = sigma, nev = levels, kw...)
#     return (real.(energies), states)
# end

resolve_degeneracies!(energies, states, vfunc::Missing, kn, degtol) = nothing
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

# function hasdegeneracies(energies, degtol)
#     has = false
#     for i in eachindex(energies), j in (i+1):length(energies)
#         if abs(energies[i] - energies[j]) < degtol
#             has = true
#             break
#         end
#     end
#     return has
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


# #######################################################################
# # BandSampling
# #######################################################################

# struct BandSampling{T<:Real,L}
#     energies::Matrix{T}
#     nenergies::Int
#     states::Array{Complex{T},3}
#     statelength::Int
#     points::Vector{SVector{L,T}}
#     npoints::Int
#     bufferstate::Vector{Complex{T}}
# end

# function BandSampling(hfunc::Function, dimh, lat::Lattice{T,E,L}, vfunc; levels = missing, degtol = sqrt(eps()), randomshift = missing, kw...) where {T,E,L}
#     shift = randomshift === missing ? zero(SVector{E,T}) : randomshift * rand(SVector{E,T})
#     points = lat.sublats[1].sites
#     npoints = length(points)

#     nenergies = levels === missing ? dimh : 2
#     statelength = dimh

#     preallocH = Matrix{Complex{T}}(undef, (dimh, dimh))

#     ordering = zeros(Int, nenergies)
#     energies = Matrix{T}(undef, (nenergies, npoints))
#     states = Array{Complex{T},3}(undef, (statelength, nenergies, npoints))

#     @showprogress "Diagonalising: " for n in 1:npoints
#         (energies_n, states_n) = spectrum(hfunc(points[n] + shift), preallocH; levels = nenergies, kw...)
#         # sort_spectrum!(energies_nk, states_nk, ordering)
#         resolve_degeneracies!(energies_n, states_n, vfunc, points[n], degtol)
#         copyslice!(energies,    CartesianIndices((1:nenergies, n:n)),
#                    energies_n, CartesianIndices(1:nenergies))
#         copyslice!(states,      CartesianIndices((1:statelength, 1:nenergies, n:n)),
#                    states_n,   CartesianIndices((1:statelength, 1:nenergies)))
#     end

#     bufferstate = zeros(Complex{T}, statelength)

#     return BandSampling(energies, nenergies, states, statelength, points, npoints, bufferstate)
# end

# function spectrum(h::SparseMatrixCSC, preallocH; levels = 2, method = missing, kw...)
#     if method === :exact || method === missing && (size(h, 1) < 129 || levels/size(h,1) > 0.2)
#         s = spectrum_dense(h, preallocH; levels = levels, kw...)
#     elseif method === :arnoldi
#         s = spectrum_arpack(h; levels = levels, kw...)
#     else
#         throw(ArgumentError("Unknown method. Choose between :arnoldi or :exact"))
#     end
#     return s
# end

# function spectrum_dense(h::SparseMatrixCSC, preallocH; levels = 2, kw...)
#     preallocH .= h
#     dimh = size(h, 1)
#     range = ((dimh - levels)÷2 + 1):((dimh + levels)÷2)
#     ee = eigen!(Hermitian(preallocH), range, kw...)
#     energies, states = ee.values, ee.vectors
#     return (energies, states)
# end

# function spectrum_arpack(h::SparseMatrixCSC; levels = 2, sigma = 1.0im, kw...)
#     (energies, states, _) = eigs(h; sigma = sigma, nev = levels, kw...)
#     return (real.(energies), states)
# end

# # function spectrum_arnoldimethods(hsparse::SparseMatrixCSC{T}, alg; nenergies = 2, kw...) where {T}
# #     h = Hermitian(hsparse)
# #     F = cholesky(H, check = false)
# #     L = ldlt!(F, H, shift = eps())
# #     map = LinearMap{T}(x -> F \ x, size(hsparse, 1))
# #     schur = partialschur(map, nev = nenergies, tol=1e-6, restarts = 100, which = LM())
# #     energies, states = partialeigen(schur)
# #     energies = 1 ./ schur.eigenvalues
# #     return (energies, states)
# # end

# # function sort_spectrum!(energies, states, ordering)
# #     if !issorted(energies)
# #         sortperm!(ordering, energies)
# #         energies .= energies[ordering]
# #         states .= states[:, ordering]
# #     end
# #     return (energies, states)
# # end

# function hasdegeneracies(energies, degtol)
#     has = false
#     for i in eachindex(energies), j in (i+1):length(energies)
#         if abs(energies[i] - energies[j]) < degtol
#             has = true
#             break
#         end
#     end
#     return has
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

# resolve_degeneracies!(energies, states, vfunc::Missing, kn, degtol) = nothing
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

# #######################################################################
# # Spectrum
# #######################################################################

# struct Spectrum{T,N,L,NL}  # E = N = L + 1 (nodes are [Blochphases..., energy])
#     bands::Mesh{T,N,L,N,NL}
#     states::Matrix{Complex{T}}
#     nenergies::Int
#     npoints::Int
# end

# Base.show(io::IO, bs::Spectrum{T,N,L}) where {T,N,L} =
#     print(io, "Spectrum{$T,$N,$L}: type $T for $L-dimensional lattice
#     Number of k-points    : $(bs.npoints)
#     Number of eigenvalues : $(bs.nenergies)
#     Size of state vectors : $(size(bs.states, 1))")

# Spectrum(sys::System; uniform = false, partitions = 5, kw...) =
#     Spectrum(sys, Brillouin(sys.lattice; uniform = uniform, partitions = partitions); kw...)
# Spectrum(sys::System{E,L}, brillouin::Brillouin{T,L}; kw...) where {E,L} =
#     Spectrum(kn -> hamiltonian!(sys, kn = kn), hamiltoniandim(sys), brillouin.lattice;
#                   velocity = (kn, axis) -> velocity!(sys, kn = kn, axis = axis), kw...)
# Spectrum(hfunc::Function, lat::Lattice{T,E}; kw...) where {T,E} =
#     Spectrum(hfunc, hamiltoniandim(hfunc(zero(SVector{E,T}))), lat; kw...)
# Spectrum(hfunc::Function, ranges::AbstractVector...; kw...) =
#     Spectrum(hfunc, cartesianlattice(ranges...); kw...)


# function Spectrum(hfunc::Function, hdim, lat::Lattice; velocity = missing, linkthreshold = 0.5, kw...)
#     isunlinked(lat) && throw(ErrorException("The band sampling lattice is not linked"))
#     bandsampling = BandSampling(hfunc, hdim, lat, velocity; kw...)
#     bandslat = bandslattice(lat, bandsampling, linkthreshold)
#     states = reshape(bandsampling.states, bandsampling.statelength, :)
#     bandsmesh = Mesh(bandslat)
#     return Spectrum(bandsmesh, states, bandsampling.nenergies, bandsampling.npoints)
# end

# function bandslattice(lat::Lattice{T,E,L}, bandsampling, linkthreshold) where {T,E,L}
#     bands = Lattice(Sublat{T,E+1}(), Bravais(SMatrix{E+1,L,T}(I)))
#     addnodes!(bands, bandsampling)
#     for samplingilink in allilinks(lat)
#         addilink!(bands.links, samplingilink, bandsampling, linkthreshold, bands)
#     end
#     return bands
# end

# function addnodes!(bands, bandsampling)
#     bandnodes = bands.sublats[1].sites
#     for nk in 1:bandsampling.npoints, ne in 1:bandsampling.nenergies
#         push!(bandnodes, vcat(bandsampling.points[nk], bandsampling.energies[ne, nk]))
#     end
#     return bands
# end

# function addilink!(bandlinks::Links, samplingilink::Ilink, sp::BandSampling, linkthreshold, bands)
#     bandnodes = bands.sublats[1,1].sites
#     dist = bravaismatrix(bands) * samplingilink.ndist
#     linearindices = LinearIndices(sp.energies)
#     state = sp.bufferstate
#     states = sp.states

#     slinkbuilder = SparseMatrixBuilder(bands, 1, 1)
#     neighiter = NeighborIterator(samplingilink, 1, (1,1))

#     @showprogress "Linking bands: " for nk_src in 1:sp.npoints, ne_src in 1:sp.nenergies
#         n_src = linearindices[ne_src, nk_src]
#         r1 = bandnodes[n_src]
#         copyslice!(state,  CartesianIndices(1:sp.statelength),
#                    states, CartesianIndices((1:sp.statelength, ne_src:ne_src, nk_src:nk_src)))
#         @inbounds for nk_target in neighbors!(neighiter, nk_src)
#             ne_target = findmostparallel(state, states, nk_target, linkthreshold)
#             if !iszero(ne_target)
#                 n_target = linearindices[ne_target, nk_target]
#                 r2 = bandnodes[n_target] + dist
#                 pushtocolumn!(slinkbuilder, n_target, _rdr(r1, r2))
#             end
#         end
#         finalisecolumn!(slinkbuilder)
#     end
#     push!(bandlinks, Ilink(samplingilink.ndist, fill(Slink(sparse(slinkbuilder)), 1, 1)))
#     return bandlinks
# end

# function findmostparallel(state::Vector{Complex{T}}, states, ktarget, linkthreshold) where {T}
#     maxproj = T(linkthreshold)
#     dotprods = abs.(state' * view(states, :, :, ktarget))
#     proj, ne = findmax(dotprods)
#     return proj > maxproj ? ne[2] : 0
# end
