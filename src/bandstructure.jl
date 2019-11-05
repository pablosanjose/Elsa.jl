#######################################################################
# Spectrum and Bandstructure
#######################################################################
# struct Spectrum{M,T}
#     states::Matrix{M}
#     energies::Vector{T}
# end

struct Band{M,A<:AbstractVector{M},MD<:Mesh}
    mesh::MD    # Mesh with missing vertices removed
    states::A   # Must be resizeable container to build & refine band
    dimstates::Int
end

function Band{M}(mesh::Mesh, dimstates::Int) where {M}
    nk = nvertices(mesh)
    states = M[]
    return Band(mesh, states, dimstates)
end

struct Bandstructure{D,M,B<:Band{M},MD<:Mesh{D}}   # D is dimension of parameter space
    bands::Vector{B}
    kmesh::MD
end

function Base.show(io::IO, b::Bandstructure{D,M}) where {D,M}
    ioindent = IOContext(io, :indent => string("  "))
    print(io,
"Bandstructure: bands for a $(D)D hamiltonian
  Bands        : $(length(b.bands))
  Element type : $(displayelements(M))")
    print(ioindent, "\n", b.kmesh)
end

# API #

# function vertices(b::Band{<:Any,T,D}) where {T,D} =
#     Union{Missing,SVector{D+1,T}}[e === missing ? missing : SVector(Tuple(v)..., e)
#         for (v, e) in zip(vertices(b.mesh), b.energies)]

# function vertmesh(b::Band{<:Any,T,D}) where {T,D}
#     verts = filter(v -> v !== missing, vertices(b.mesh))
#     faces = SVector{D+1,SVector{D+1,real(T)}}[]
#     sizehint!(faces, length(b.mesh.simplices))
#     kverts = vertices(b.mesh)
#     for simplex in b.mesh.simplices
#         any(v -> b.energies[v] === missing, simplex) && continue
#         face = _face(simplex, kverts, b.energies)
#         push!(faces, face)
#     end
#     return faces
# end

# function simplices(b::Band{<:Any,T,D}) where {T,D}
#     faces = SVector{D+1,SVector{D+1,real(T)}}[]
#     sizehint!(faces, length(b.mesh.simplices))
#     kverts = vertices(b.mesh)
#     for simplex in b.mesh.simplices
#         any(v -> b.energies[v] === missing, simplex) && continue
#         face = _face(simplex, kverts, b.energies)
#         push!(faces, face)
#     end
#     return faces
# end

# _face(simplex, kverts, energies) = (v -> _vertex(kverts[v], energies[v])).(simplex)
# _vertex(kv::SVector{D,T}, ϵ) where {D,T} = SVector{D+1,T}(Tuple(kv)..., T(real(ϵ)))

#######################################################################
# bandstructure
#######################################################################
bandstructure(h::Hamiltonian{<:Any,L,M}, resolution::Integer = 13; kw...) where {L,M} =
    bandstructure!(diagonalizer(h; kw...), h,  marchingmesh(h, resolution))
    # barrier for type-unstable diagonalizer

function bandstructure!(d::Diagonalizer, h::Hamiltonian{<:Lattice,<:Any,M}, mesh::MD) where {M,D,T,MD<:Mesh{D,T}}
    nϵ = d.levels
    dimh = size(h, 1)
    nk = nvertices(mesh)
    ϵks = Matrix{T}(undef, nϵ, nk)
    ψks = Array{M,3}(undef, dimh, nϵ, nk)
    p = Progress(nk, "Step 1/2 - Diagonalising: ")
    for (n, ϕs) in enumerate(vertices(mesh))
        bloch!(d.matrix, h, ϕs)
        (ϵk, ψk) = diagonalize(d)
        resolve_degeneracies!(ϵk, ψk, d, ϕs)
        copyslice!(ϵks, CartesianIndices((1:nϵ, n:n)),
                   ϵk,  CartesianIndices((1:nϵ,)))
        copyslice!(ψks, CartesianIndices((1:dimh, 1:nϵ, n:n)),
                   ψk,  CartesianIndices((1:dimh, 1:nϵ)))
        ProgressMeter.next!(p; showvalues = ())
    end

    bands = Band{M,Vector{M},Mesh{D+1,T,Vector{SVector{D+1,T}},Vector{NTuple{D+1,Int}}}}[]
    bandindices = Vector{Int}(undef, nk) # the 1:nϵ index for each k point. 0 == missing
    p = Progress(nϵ, "Step 2/2 - Connecting bands: ")
    for nb in 1:nϵ
        findbandindices!(bandindices, nb, ψks, mesh, d.minprojection)
        # fill!(bandindices, nb)
        band = extractband(bandindices, ϵks, ψks, mesh)
        push!(bands, band)
        ProgressMeter.next!(p; showvalues = ())
    end
    return Bandstructure(bands, mesh)
end

function findbandindices!(bandindices, nb, ψks, mesh, minprojection)
    dimh, nϵ, nk = size(ψks)
    fill!(bandindices, 0)
    bandindices[1] = nb
    for srck in 1:nk, edgek in edges(mesh, srck)
        destk = edgedest(mesh, edgek)
        srcb = bandindices[srck]
        proj, destb = findmostparallel(ψks, destk, srcb, srck)
        if proj > minprojection
            if iszero(bandindices[destk])
                bandindices[destk] = destb
            elseif bandindices[destk] != destb
                throw(error("Non-trivial band degeneracy detected. Resolution not yet implemented."))
            end
            # bandindices[destk] = destb
        end
    end
    return bandindices
end

function findmostparallel(ψks::Array{M,3}, destk, srcb, srck) where {M}
    T = real(eltype(M))
    dimh, nϵ, nk = size(ψks)
    maxproj = zero(T)
    destb = 0
    srcb == 0 && return maxproj, destb
    @inbounds for nb in 1:nϵ
        proj = zero(M)
        @simd for i in 1:dimh
            proj += ψks[i, nb, destk]' * ψks[i, srcb, srck]
        end
        absproj = T(abs(tr(proj)))
        if maxproj < absproj
            destb = nb
            maxproj = absproj
        end
    end
    return maxproj, destb
end

function extractband(bandindices, ϵks, ψks, mesh::Mesh{D,T}) where {D,T}
    dimh, nϵ, nk = size(ψks)
    states = similar(ψks, dimh * nk)
    vertices = Vector{SVector{D+1,T}}(undef, nk)
    k´ = 0
    for (k, ind) in enumerate(bandindices)
        if !iszero(ind)
            k´ += 1
            vertices[k´] = SVector(Tuple(mesh.vertices[k])..., ϵks[ind, k])
            copyto!(states, 1 + dimh * (k´ - 1), ψks, 1 + dimh * (k - 1), dimh)
            bandindices[k] = k´ # Reuse to store new vertex indices
        end
    end
    if k´ < nk
        resize!(vertices, k´)
        resize!(states, k´ * dimh)
        simplices = extractsimplices(mesh.simplices, bandindices)
        adjmat = extractsadjacencies(mesh.adjmat, bandindices)
    else
        simplices = copy(vec(mesh.simplices))
        adjmat = copy(mesh.adjmat)
    end
    mesh´ = Mesh(vertices, adjmat, simplices)
    band = Band(mesh´, states, dimh)
    return band
end

function extractsimplices(simplices::AbstractVector{NTuple{N,Int}}, indices) where {N}
    simplices´ = similar(simplices)
    n = 0
    for simp in simplices
        simp´ = ntuple(i -> indices[simp[i]], Val(N))
        if all(!iszero, simp´)
            n += 1
            simplices´[n] = simp´
        end
    end
    resize!(simplices´, n)
    return simplices´
end

## This is simpler, but allocates more, and is slower
# extractsadjacencies(adjmat, bandindices) =
#     adjmat[(!iszero).(bandindices), (!iszero).(bandindices)]

function extractsadjacencies(adjmat::AbstractSparseMatrix{Tv}, bandindices) where {Tv}
    n = count(!iszero, bandindices)
    b = SparseMatrixBuilder{Tv}(n, n)
    for col in 1:size(adjmat, 2)
        iszero(bandindices[col]) && continue
        for ptr in nzrange(adjmat, col)
            row = rowvals(adjmat)[ptr]
            iszero(bandindices[row]) || pushtocolumn!(b, row, nonzeros(adjmat)[ptr])
        end
        finalizecolumn!(b)
    end
    adjmat´ = sparse(b)
    return adjmat´
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
#         finalizecolumn!(slinkbuilder)
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
