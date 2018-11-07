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
    # spectrum_fake(h)
end

# spectrum_fake(h::SparseMatrixCSC; kw...) = (rand(size(h,1)), rand(size(h)...))

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
    bandmeshlat = emptybandmeshlat(bz)
    addnodes!(bandmeshlat, spectrum)
    for (meshilink, bzilink) in zip(allilinks(bandmeshlat), allilinks(bzmeshlat))
        link!(meshilink, bzilink, spectrum, linkthreshold, bandmeshlat)
    end
    states = reshape(spectrum.states, spectrum.statelength, :)
    bandmesh = Mesh(bandmeshlat)
    return Bandstructure(bandmesh, states, spectrum.nenergies, spectrum.npoints)
    # bandmeshlat
end

function emptybandmeshlat(bz::BrillouinMesh{T,L}) where {T,L}
    bandmeshlat = Lattice(Sublat{T,L+1}(), Bravais(SMatrix{L+1,L,T}(I)))
    bandmeshlat.links.interlinks = 
        [emptyilink(ilink.ndist, bandmeshlat.sublats) for ilink in bz.mesh.lattice.links.interlinks]
    return bandmeshlat
end

function addnodes!(bandmeshlat, spectrum)
    meshnodes = bandmeshlat.sublats[1].sites
    for nk in 1:spectrum.npoints, ne in 1:spectrum.nenergies
        push!(meshnodes, vcat(spectrum.knpoints[nk], spectrum.energies[ne, nk]))
    end
    return bandmeshlat
end

function link!(meshilink::Ilink{T,L1,L}, bzilink, sp::Spectrum, linkthreshold, bandmesh) where {T,L1,L}
    meshnodes = bandmesh.sublats[1,1].sites
    dist = bravaismatrix(bandmesh) * meshilink.ndist
    linearindices = LinearIndices(sp.energies)
    state = sp.bufferstate
    states = sp.states
    srcpointers = meshilink.slinks[1,1].srcpointers
    targets = meshilink.slinks[1,1].targets
    rdr = meshilink.slinks[1,1].rdr
    length(targets) == length(rdr) == length(srcpointers) - 1 == 0 || throw("Bug in linkbandmesh!")
    
    counter = 1
    @showprogress "Linking bands: " for nk_src in 1:sp.npoints, ne_src in 1:sp.nenergies
        n_src = linearindices[ne_src, nk_src]
        copyslice!(state,  CartesianIndices(1:sp.statelength), 
                   states, CartesianIndices((1:sp.statelength, ne_src:ne_src, nk_src:nk_src)))
        @inbounds for nk_target in neighbors(bzilink, nk_src, (1,1))
            ne_target = findmostparallel(state, states, nk_target, linkthreshold)
            if !iszero(ne_target)
                n_target = linearindices[ne_target, nk_target]
                push!(targets, n_target)
                push!(rdr, _rdr(meshnodes[n_src], meshnodes[n_target] + dist))
                counter += 1
            end
        end
        push!(srcpointers, counter)
    end
    return meshilink
end
   
function findmostparallel(state::Vector{Complex{T}}, states, ktarget, linkthreshold) where {T}
    maxproj = T(linkthreshold)
    dotprods = abs.(state' * view(states, :, :, ktarget))
    proj, ne = findmax(dotprods)
    return proj > maxproj ? ne[2] : 0
end