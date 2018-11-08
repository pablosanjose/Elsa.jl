
#######################################################################
# BrillouinMesh
#######################################################################
"""
    BrillouinMesh(lat::Lattice; uniform::Bool = false, partitions = 5)

Discretizes the Brillouin zone of Lattice `lat` into hypertriangular finite 
elements, using a certain number of `partitions` per Bravais axis (accepts 
an integer or a tuple of integers, one per axis). Keyword `uniform` specifies 
the type of mesh, either "uniform" (as close to equilateral as possible) or 
"simple" (cartesian partition)

# Examples
```jldoctest
julia> BrillouinMesh(Lattice(:honeycomb), uniform = true, partitions = 200)
BrillouinMesh{Float64,2} : discretization of 2-dimensional Brillouin zone
    Mesh type  : uniform
    Vertices   : 40000 
    Partitions : (200, 200)
    3-elements : 80000
```
"""
struct BrillouinMesh{T,L,N,LL}
    mesh::Mesh{T,L,L,N,LL}
    uniform::Bool
    partitions::NTuple{L,Int}
end

function BrillouinMesh(lat::Lattice{T,E,L}; uniform::Bool = false, partitions = 5) where {T,E,L}
    partitions_tuple = tontuple(Val(L), partitions)
    if uniform
        meshlat = uniform_mesh(lat, partitions_tuple)
    else
        meshlat = simple_mesh(lat, partitions_tuple)
    end
    # wrappedmesh = wrap(mesh)
    mesh = Mesh(meshlat)
    return BrillouinMesh(mesh, uniform, partitions_tuple)
end
BrillouinMesh(sys::System; kw...) = BrillouinMesh(sys.lattice; kw...)

Base.show(io::IO, m::BrillouinMesh{T,L,N}) where {T,L,N} =
    print(io, "BrillouinMesh{$T,$L} : discretization of $L-dimensional Brillouin zone
    Mesh type  : $(m.uniform ? "uniform" : "simple")
    Vertices   : $(nsites(m.mesh.lattice)) 
    Partitions : $(m.partitions)
    $N-elements : $(nelements(m))")

nelements(m::BrillouinMesh) = nelements(m.mesh)

# phi-space sampling z, k-space G'z. M = diagonal(partitions)
# M G' z =  Tr * n, where n are SVector{L,Int}, and Tr is a hypertriangular lattice
# For some integer S = (n1,n2...), (z1, z2, z3) = I (corners of BZ).
# Hence S = round.(Tr^{-1} G' M) = supercell. Bravais are z_i for n = I, so simply S^{-1}
# Links should be fixed at the Tr level, then transform so that D * Tr = S^{-1}, and do Supercell(S)
function uniform_mesh(lat::Lattice{T,E,L}, partitions_tuple::NTuple{L,Int}) where {T,E,L}
    M = diagsmatrix(partitions_tuple)
    A = qr(bravaismatrix(lat)).R
    Gt = qr(transpose(inv(A))).R
    Gt = Gt / Gt[1,1]
    Tr = hypertriangular(lat)
    S = round.(Int, inv(Tr) * Gt * M)
    iS = inv(S)
    D = iS * inv(Tr)
    meshlat = Lattice(Sublat(zero(SVector{L,T})), Bravais(Tr), LinkRule(1))
    meshlat = transform!(meshlat, r -> D * r)
    meshlat = lattice!(meshlat, Supercell(S))
    # methlat = transform!(meshlat, r -> Gt * r)  # to go back to k space
    return meshlat
end

# Transformation D takes hypertriangular to minisquare (phi-space delta): D * Tr = M^{-1}
# However, Tr has to be chosen to match bravais angles of G' as much as possible: Trsigned
# Build+link Trsigned, transform, Supercell(M)
function simple_mesh(lat::Lattice{T,E,L}, partitions_tuple::NTuple{L,Int}) where {T,E,L}
    M = diagsmatrix(partitions_tuple)
    A = qr(bravaismatrix(lat)).R
    Gt = qr(transpose(inv(A))).R
    Gt = Gt / Gt[1,1]
    Tr = hypertriangular(lat)
    Trsigned = SMatrix{L,0,T}()
    for i in 1:L
        Trsigned = hcat(Trsigned, Tr[:,i] * sign_positivezero(Gt[1,i]))
    end
    D = inv(Trsigned * M)
    meshlat = Lattice(Sublat(zero(SVector{L,T})), Bravais(Trsigned), LinkRule(1))
    meshlat = transform!(meshlat, r -> D * r)
    meshlat = lattice!(meshlat, Supercell(M))
    return meshlat
end

hypertriangular(lat::Lattice{T,E,L}) where {T,E,L} = hypertriangular(SMatrix{L,1,T}(I))
function hypertriangular(s::SMatrix{L,L2,T}) where {L,L2,T} 
    v1 = s[:,L2]
    factor = T(1/(L2+1))
    v2 = modifyat(v1, L2, v1[L2]*factor)
    v2 = modifyat(v2, L2 + 1, v1[L2]*sqrt(1 - factor^2))
    return hypertriangular(hcat(s, v2))
end
hypertriangular(s::SMatrix{L,L}) where L = s

#######################################################################
# Spectrum
#######################################################################

struct Spectrum{T<:Real,L}
    energies::Matrix{T}
    nenergies::Int
    states::Array{Complex{T},3}
    statelength::Int
    knpoints::Vector{SVector{L,T}}
    npoints::Int
    bufferstate::Vector{Complex{T}}
end

function Spectrum(sys::System{T,E,L}, bzmesh::BrillouinMesh; kw...) where {T,E,L}
    # shift = 0.02 .+ zero(SVector{E,T})
    shift = 0.02 * rand(SVector{E,T})
    knpoints = bzmesh.mesh.lattice.sublats[1].sites
    npoints = length(knpoints)
    first_h = hamiltonian(sys, kn = knpoints[1] + shift)
    buffermatrix = Matrix{Complex{T}}(undef, size(first_h))
    (energies_kn, states_kn) = spectrum(first_h, buffermatrix; kw...)
    (statelength, nenergies) = size(states_kn)
    
    energies = Matrix{T}(undef, (nenergies, npoints))
    states = Array{Complex{T},3}(undef, (statelength, nenergies, npoints))
    copyslice!(energies,    CartesianIndices((1:nenergies, 1:1)), 
               energies_kn, CartesianIndices(1:nenergies))
    copyslice!(states,      CartesianIndices((1:statelength, 1:nenergies, 1:1)), 
               states_kn,   CartesianIndices((1:statelength, 1:nenergies)))

    @showprogress "Diagonalising: " for nk in 2:npoints
        (energies_nk, states_nk) = spectrum(hamiltonian(sys, kn = knpoints[nk] + shift), buffermatrix; kw...)
        copyslice!(energies,    CartesianIndices((1:nenergies, nk:nk)), 
                   energies_nk, CartesianIndices(1:nenergies))
        copyslice!(states,      CartesianIndices((1:statelength, 1:nenergies, nk:nk)), 
                   states_nk,   CartesianIndices((1:statelength, 1:nenergies)))
    end
    
    bufferstate = zeros(Complex{T}, statelength)
    
    return Spectrum(energies, nenergies, states, statelength, knpoints, npoints, bufferstate)
end

function spectrum(h::SparseMatrixCSC, buffermatrix; levels = missing, kw...)
    if ismissing(levels) || size(h, 1) < 129 || levels/size(h,1) > 0.2
        return spectrum_dense(h, buffermatrix; kw...)
    else
        return spectrum_arpack(h; kw...)
    end
end

function spectrum_dense(h::SparseMatrixCSC, buffermatrix; levels = missing, kw...)
    buffermatrix .= h
    dimh = size(h, 1)
    range = ismissing(levels) ? (1:dimh) : (((dimh - levels)÷2 + 1):((dimh + levels)÷2))
    ee = eigen(Hermitian(buffermatrix), range)
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

#######################################################################
# Bandstructure
#######################################################################

struct Bandstructure{T,N,L,NL}  # E = N = L + 1 (nodes are [Blochphases..., energy])
    mesh::Mesh{T,N,L,N,NL}
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
    Bandstructure(sys, BrillouinMesh(sys.lattice; uniform = uniform, partitions = partitions); kw...) 

function Bandstructure(sys::System{T,E,L}, bz::BrillouinMesh{T,L}; linkthreshold = 0.5, kw...) where {T,E,L}
    spectrum = Spectrum(sys, bz; kw...)
    bzmeshlat = bz.mesh.lattice
    bmeshlat = bandmeshlat(bz, spectrum, linkthreshold)
    states = reshape(spectrum.states, spectrum.statelength, :)
    bandmesh = Mesh(bmeshlat)
    return Bandstructure(bandmesh, states, spectrum.nenergies, spectrum.npoints)
end

function bandmeshlat(bz::BrillouinMesh{T,L}, spectrum, linkthreshold) where {T,L}
    bandmeshlat = Lattice(Sublat{T,L+1}(), Bravais(SMatrix{L+1,L,T}(I)))
    addnodes!(bandmeshlat, spectrum)
    bzmeshlat = bz.mesh.lattice
    bzmeshlinks = bzmeshlat.links
    bandmeshlat.links.intralink = emptyilink(bzmeshlinks.intralink.ndist, bandmeshlat.sublats)
    bandmeshlat.links.interlinks = 
        [emptyilink(ilink.ndist, bandmeshlat.sublats) for ilink in bzmeshlinks.interlinks]
    for (bmeshilink, bzmeshilink) in zip(allilinks(bandmeshlat), allilinks(bzmeshlat))
        linkbands!(bmeshilink, bzmeshilink, spectrum, linkthreshold, bandmeshlat)
    end
    return bandmeshlat
end

function addnodes!(bandmeshlat, spectrum)
    meshnodes = bandmeshlat.sublats[1].sites
    for nk in 1:spectrum.npoints, ne in 1:spectrum.nenergies
        push!(meshnodes, vcat(spectrum.knpoints[nk], spectrum.energies[ne, nk]))
    end
    return bandmeshlat
end

function linkbands!(meshilink::Ilink{T,L1,L}, bzilink, sp::Spectrum, linkthreshold, bandmesh) where {T,L1,L}
    meshnodes = bandmesh.sublats[1,1].sites
    dist = bravaismatrix(bandmesh) * meshilink.ndist
    linearindices = LinearIndices(sp.energies)
    state = sp.bufferstate
    states = sp.states
    
    slink = meshilink.slinks[1,1]
    #emptyslink(bandmesh, 1, 1)
    counter = 1
    column = 0
    @showprogress "Linking bands: " for nk_src in 1:sp.npoints, ne_src in 1:sp.nenergies
        column += 1
        slink.rdr.colptr[column] = counter
        n_src = linearindices[ne_src, nk_src]
        copyslice!(state,  CartesianIndices(1:sp.statelength), 
                   states, CartesianIndices((1:sp.statelength, ne_src:ne_src, nk_src:nk_src)))
        @inbounds for nk_target in neighbors(bzilink, nk_src, (1,1))
            ne_target = findmostparallel(state, states, nk_target, linkthreshold)
            if !iszero(ne_target)
                n_target = linearindices[ne_target, nk_target]
                unsafe_pushlink!(slink, n_src, n_target, _rdr(meshnodes[n_src], meshnodes[n_target] + dist))
                counter += 1
            end
        end
    end
    slink.rdr.colptr[end] = counter
    # meshilink.slinks[1,1] = slink
    return meshilink
end

# function link!(meshilink::Ilink{T,L1,L}, bzilink, sp::Spectrum, linkthreshold, bandmesh) where {T,L1,L}
#     meshnodes = bandmesh.sublats[1,1].sites
#     dist = bravaismatrix(bandmesh) * meshilink.ndist
#     linearindices = LinearIndices(sp.energies)
#     state = sp.bufferstate
#     states = sp.states
#     srcpointers = meshilink.slinks[1,1].srcpointers
#     targets = meshilink.slinks[1,1].targets
#     rdr = meshilink.slinks[1,1].rdr
#     length(targets) == length(rdr) == length(srcpointers) - 1 == 0 || throw("Bug in linkbandmesh!")
    
#     counter = 1
#     @showprogress "Linking bands: " for nk_src in 1:sp.npoints, ne_src in 1:sp.nenergies
#         n_src = linearindices[ne_src, nk_src]
#         copyslice!(state,  CartesianIndices(1:sp.statelength), 
#                    states, CartesianIndices((1:sp.statelength, ne_src:ne_src, nk_src:nk_src)))
#         @inbounds for nk_target in neighbors(bzilink, nk_src, (1,1))
#             ne_target = findmostparallel(state, states, nk_target, linkthreshold)
#             if !iszero(ne_target)
#                 n_target = linearindices[ne_target, nk_target]
#                 push!(targets, n_target)
#                 push!(rdr, _rdr(meshnodes[n_src], meshnodes[n_target] + dist))
#                 counter += 1
#             end
#         end
#         push!(srcpointers, counter)
#     end
#     return meshilink
# end
   
function findmostparallel(state::Vector{Complex{T}}, states, ktarget, linkthreshold) where {T}
    maxproj = T(linkthreshold)
    dotprods = abs.(state' * view(states, :, :, ktarget))
    proj, ne = findmax(dotprods)
    return proj > maxproj ? ne[2] : 0
end