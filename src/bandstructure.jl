#######################################################################
# Bandstructure
#######################################################################

struct Bandstructure{T,N}  # N = L + 1 (nodes are energy-Blochphases)
    mesh::Lattice{T,N,0,0}
    states::Matrix{T}
    elements::Elements{T,N}
end

function Bandstructure(sys::System{T,E,L}, bzmesh::BrillouinMesh{T,L}; threshold = 0.3, kw...) where {T,E,L}
    (energies, states) = spectrum(sys, bzmesh; kw...)
    statelength = size(states, 1)
    kpoints = bzmesh.mesh.sublats[1].sites
    nevals, nkpoints = size(energies)
    nkpoints == length(kpoints) || throw("k-point mismatch (should not happen)")
    
    mesh = Lattice(Sublat{T,L+1}())
    meshnodes = mesh.sublats[1].sites
    for nk in 1:nkpoints, ne in 1:nevals
        push!(meshnodes, vcat(energies[ne, nk], kpoints[nk])
    end

    linkI = Int[]
    linkJ = Int[]
    linkV = Tuple{SVector{L+1,T}, SVector{L+1,T}}[]
    slink_bzmesh = bzmesh.mesh.links.intralink.slink[1,1]
    linearindices = LinearIndices(energies)
    state = Vector{Complex{T}}(undef, statelength)
    for nk_src in 1:nkpoints
        for ne_src in 1:nevals
            copyto!(state,  CartesianIndices(1:statelength), 
                    states, CartesianIndices((1:statelength, ne_src:ne_src, nk_src:nk_src)))
            for nk_target in neighbors(slink_bzmesh, nk_src)
                ne_target = findmostparallel(state, states, nk_target, threshold)
                if !iszero(ne_target)
                    n_src = linearindices[ne_src, nk_src]
                    n_target = linearindices[ne_target, nk_target]
                    rdr = _rdr(meshnodes[n_src], meshnodes[n_target])
                    push!(linkI, n_target)
                    push!(linkJ, n_src)
                    push!(linkV, rdr)
                end
            end
        end
    end
    sp = sparse(linkI, linkJ, linkV)
    mesh.links.intralink.slinks[1,1] = Slink(rowvals(sp), sp.colptr, nonzeros(sp))
    elements = Elements(mesh)
    return Bandstructure(mesh, reshape!(states, statelength, :), elements)
end

function findmostparallel(state::Vector{Complex{T}}, states, ktarget, threshold) where {CT}
    target = 0
    maxproj = T(threshold)
    for ne in axis(states, 2)
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

function spectrum(sys::System{T}, bzmesh::BrillouinMesh; kw...)
    kpoints = bzmesh.mesh.sublats[1].sites
    nkpoints = length(kpoints)
    (energies_kn, states_kn) = spectrum(hamiltonian(sys, kn = knpoints[1]); kw...)
    (statelength, nevals) = size(states_kn)
    
    energies = Matrix{T}(undef, (nevals, nkpoints))
    states = Array{T,3}(undef, (statelength, nevals, nkpoints))
    copyto!(energies,    CartesianIndices((1:nevals, 1:1)), 
            energies_kn, CartesianIndices(1:nevals))
    copyto!(states,      CartesianIndices((1:statelength, 1:nevals, 1:1)), 
            energies_kn, CartesianIndices(1:statelength, 1:nevals))

    for kn in 2:kpoints
        (energies_kn, states_kn) = spectrum(hamiltonian(sys, kn = kn); kw...)
        copyto!(energies,    CartesianIndices((1:nevals, kn:kn)), 
                energies_kn, CartesianIndices(1:nevals))
        copyto!(states,      CartesianIndices((1:statelength, 1:nevals, kn:kn)), 
                energies_kn, CartesianIndices(1:statelength, 1:nevals))
    end
    
    return (energies, states)
end

function spectrum(h::SparseMatrixCSC; kw...)
    (energies, states) = eigs(h; kw...)
    return (re.(energies), states)
end
