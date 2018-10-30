#######################################################################
# Bandstructure
#######################################################################

struct Subband{T,N}  # N = L + 1 (nodes are energy-Blochphases)
    mesh::Lattice{T,N,0,0}
    states::Matrix{T}
    elements::Elements{T,N}
end

struct Bandstructure{T,N} 
    subbands::Vector{Subband{T,N}}
end

function Bandstructure(sys::System{T,E,L}, bzm::BrillouinMesh{T,L}; kw...) where {T,E,L}
    (energies, states) = spectrum(sys, bzm; kw...)
    nenergies = size(energies, 1)
    subband_inds = zeros(Int, size(energies))
    for element in bzm.elements
        classifystates!(subband_inds, element, states)
    end
    nsubbands = maximum(subband_inds)
    sbmeshes = [Lattice(Sublat{T,L+1}()) for i in 1:nsubbands]
    ksites = bzm.mesh.sublats[1].sites
    for ci in CartesianIndices(subband_inds)
        (ei, ki) = Tuple(ci)
        sb = subband_inds[ci]
        iszero(sb) || push!(sbmeshes[sb].sublats[1].sites, vcat(energies[ci], ksites[ki])
        # How do we link?? sbmesh does not have all the sites of bzm.mesh
    end
    sbstates = [filterstates(states, i, subbands_inds) for i in 1:nsubbands]
    subbands = [Subband(sbmeshes[i], sbstates[i], Elements(sbmeshes[i])) for i in 1:nsubbands]
    bandstructure = Bandsstructure(subbands)
    return bandstructure
end
