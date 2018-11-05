
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
    # knpoints = bzmesh.mesh.sublats[1].sites
    shift = 0.01*rand(SVector{E,T})
    knpoints = bzmesh.mesh.sublats[1].sites
    npoints = length(knpoints)
    first_h = hamiltonian(sys, kn = knpoints[1]+shift)
    buffermatrix = Matrix{Complex{T}}(undef, size(first_h))
    (energies_kn, states_kn) = spectrum(first_h, buffermatrix; kw...)
    (statelength, nenergies) = size(states_kn)
    
    energies = Matrix{T}(undef, (nenergies, npoints))
    states = Array{Complex{T},3}(undef, (statelength, nenergies, npoints))
    copyslice!(energies,    CartesianIndices((1:nenergies, 1:1)), 
               energies_kn, CartesianIndices(1:nenergies))
    copyslice!(states,      CartesianIndices((1:statelength, 1:nenergies, 1:1)), 
               states_kn,   CartesianIndices((1:statelength, 1:nenergies)))

    for nk in 2:npoints
        (energies_nk, states_nk) = spectrum(hamiltonian(sys, kn = knpoints[nk] + shift), buffermatrix; kw...)
        copyslice!(energies,    CartesianIndices((1:nenergies, nk:nk)), 
                   energies_nk, CartesianIndices(1:nenergies))
        copyslice!(states,      CartesianIndices((1:statelength, 1:nenergies, nk:nk)), 
                   states_nk,   CartesianIndices((1:statelength, 1:nenergies)))
    end
    
    bufferstate = zeros(Complex{T}, statelength)
    
    return Spectrum(energies, nenergies, states, statelength, knpoints, npoints, bufferstate)
end

function spectrum(h::SparseMatrixCSC, buffermatrix; kw...)
    if size(h, 1) < 8
        return spectrum_dense(h, buffermatrix; kw...)
    else
        return spectrum_arpack(h; kw...)
    end
end

function spectrum_dense(h::SparseMatrixCSC, buffermatrix; kw...)
    buffermatrix .= h
    ee = eigen(buffermatrix)
    energies, states = ee.values, ee.vectors
    return (real.(energies), states)
end

function spectrum_arpack(h::SparseMatrixCSC; levels = 2, sigma = 1.0im, kw...)
    (energies, states, _) = eigs(h; sigma = sigma, nev = levels, kw...)
    return (real.(energies), states)
end


# function spectrum(hsparse::SparseMatrixCSC{T}, alg; nenergies = 2, kw...) where {T}
#     h = Hermitian(hsparse)
#     F = cholesky(H, check = false)
#     L = ldlt!(F, H, shift = eps())
#     map = LinearMap{T}(x -> F \ x, size(hsparse, 1))
#     schur = partialschur(map, nev = nenergies, tol=1e-6, restarts = 100, which = LM())
#     energies = 1 ./ schur.eigenvalues
#     states = schur.Q
#     return (energies, states)
# end

# function spectrum_dense(h::SparseMatrixCSC{T}; energies = missing, kw...) where {T}
#     ismissing(energies) && nev = 
#     ee = eigs(h; sigma = 1.0im, kw...)
#     (energies, states) = (ee.values, ee.vectors)
#     return (real.(energies), states)
# end

#######################################################################
# Bandstructure
#######################################################################

struct Bandstructure{T,N,L,NL}  # N = L + 1 (nodes are energy-Blochphases)
    mesh::Lattice{T,N,L,NL}
    states::Matrix{Complex{T}}
    elements::Elements{N}
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

function Bandstructure(sys::System{T,E,L}, bz::BrillouinMesh{T,L}; threshold = 0.5, kw...) where {T,E,L}
    spectrum = Spectrum(sys, bz; kw...)
    bzmesh = bz.mesh
    bandmesh = emptybandmesh(bz)
    addnodesbandmesh!(bandmesh, spectrum)
    for (meshilink, bzilink) in zip(allilinks(bandmesh), allilinks(bzmesh))
        linkbandmesh!(meshilink, bzilink, spectrum, threshold, bandmesh)
    end
    statesmatrix = reshape(spectrum.states, spectrum.statelength, :)
    elements = Elements(bandmesh, Val(L+1))
    return Bandstructure(bandmesh, statesmatrix, elements, spectrum.nenergies, spectrum.npoints)
end

function emptybandmesh(bz::BrillouinMesh{T,L}) where {T,L}
    bandmesh = Lattice(Sublat{T,L+1}(), Bravais(SMatrix{L+1,L,T}(I)))
    bandmesh.links.interlinks = 
        [emptyilink(ilink.ndist, bandmesh.sublats) for ilink in bz.mesh.links.interlinks]
    return bandmesh
end

function addnodesbandmesh!(bandmesh, spectrum)
    meshnodes = bandmesh.sublats[1].sites
    for nk in 1:spectrum.npoints, ne in 1:spectrum.nenergies
        push!(meshnodes, vcat(spectrum.knpoints[nk], spectrum.energies[ne, nk]))
    end
    return bandmesh
end

function linkbandmesh!(meshilink::Ilink{T,L1,L}, bzilink, sp::Spectrum, threshold, bandmesh) where {T,L1,L}
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
    for nk_src in 1:sp.npoints, ne_src in 1:sp.nenergies
        n_src = linearindices[ne_src, nk_src]
        copyslice!(state,  CartesianIndices(1:sp.statelength), 
                    states, CartesianIndices((1:sp.statelength, ne_src:ne_src, nk_src:nk_src)))
        for nk_target in neighbors(bzilink, nk_src, (1,1))
            ne_target = findmostparallel(state, states, nk_target, threshold)
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
   
function findmostparallel(state::Vector{Complex{T}}, states, ktarget, threshold) where {T}
    target = 0
    maxproj = T(threshold)
    for ne in axes(states, 2)
        dotprod = zero(Complex{T})
        for nphi in 1:length(state)
            dotprod += conj(state[nphi]) * states[nphi, ne, ktarget]
        end
        proj = abs(dotprod)
        if proj > maxproj
            maxproj = proj
            target = ne
        end
    end
    return target
end