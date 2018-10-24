#######################################################################
# Hamiltonian
#######################################################################
struct SparseWorkspace{T}
    klasttouch::Vector{Int}
    csrrowptr::Vector{Int}
    csrcolval::Vector{Int}
    csrnzval::Vector{Complex{T}}
end
function SparseWorkspace{T}(dimh, lengthI) where {T} 
    klasttouch = Vector{Int}(undef, dimh)
    csrrowptr = Vector{Int}(undef, dimh + 1)
    csrcolval = Vector{Int}(undef, lengthI)
    csrnzval = Vector{Complex{T}}(undef, lengthI)
    SparseWorkspace{T}(klasttouch, csrrowptr, csrcolval, csrnzval)
end

struct Hamiltonian{T,L}
    I::Vector{Int}
    J::Vector{Int}
    V::Vector{Complex{T}}
    Voffsets::Vector{Int}
    Vn::Vector{Vector{Complex{T}}}
    ndist::Vector{SVector{L, Int}}
    workspace::SparseWorkspace{T}
    matrix::SparseMatrixCSC{Complex{T},Int}
end

#######################################################################
# build hamiltonian
#######################################################################

#hamiltonian(lat::Lattice, model::Model) = hamiltonian(hamiltoniantype(lat, model), lat, model)

function hamiltonian(lat::Lattice{T,E,L}, model::Model) where {T,E,L}
    isunlinked(lat) && return missing

    dimh = hamiltoniandim(lat, model)
    
    I = Int[]
    J = Int[]
    V = Complex{T}[]
    Vn = [Complex{T}[] for k in 1:length(lat.links.interlinks)]
    Voffsets = Int[]
    
    hbloch!(I, J, V, model, lat, lat.links.intralink, false)
    push!(Voffsets, length(V) + 1)

    for (k, interlink) in enumerate(lat.links.interlinks)
        hbloch!(I, J, Vn[k], model, lat, interlink, true)
        append!(V, Vn[k])
        push!(Voffsets, length(V) + 1)
    end

    ndist = [interlink.ndist for interlink in lat.links.interlinks]
    workspace = SparseWorkspace{T}(dimh, length(I))

    mat = sparse!(I, J, V, dimh, workspace)
    
    return Hamiltonian(I, J, V, Voffsets, Vn, ndist, workspace, mat)
end

function hamiltoniantype(lat::Lattice{T,E}, model) where {T,E}
    zerovec = zero(SVector{E,T})
    if !isempty(model.onsites)
        term = first(model.onsites)(zerovec)
    elseif !(model.defonsite isa NoOnsite)
        term = model.defonsite(zerovec)
    elseif !isempty(model.hoppings)
        term = first(model.hoppings)((zerovec, zerovec))
    elseif !(model.defhopping isa NoHopping)
        term = model.defhopping((zerovec, zerovec))
    else
        term = SMatrix{0,0,T,0}()
    end
    return eltype(term)
end

function hamiltoniandim(lat, model) 
    dim = 0
    for (s, d) in enumerate(sublatdims(lat, model))
        dim += nsites(lat.sublats[s]) * d
    end
    return dim
end

sparse!(I, J, V, dimh, workspace::SparseWorkspace) = 
    sparse!(I, J, V, dimh, dimh, +, 
            workspace.klasttouch, workspace.csrrowptr, 
            workspace.csrcolval, workspace.csrnzval)

sparse!(h, I, J, V, dimh, workspace::SparseWorkspace) = 
    sparse!(I, J, V, dimh, dimh, +, 
            workspace.klasttouch, workspace.csrrowptr, 
            workspace.csrcolval, workspace.csrnzval, 
            h.colptr, h.rowval, h.nzval)

updatehamiltonian!(h::Hamiltonian) = sparse!(h.matrix, h.I, h.J, h.V, size(h.matrix, 1), h.workspace)

function hbloch!(I, J, V, model, lat, ilink, isinter)
    sdims = sublatdims(lat, model) # orbitals per site of each sublattice
    coloffsetblock = 0
    for (s1, subcols) in enumerate(sdims)
        rowoffsetblock = 0
        for (s2, subrows) in enumerate(sdims)
            if !isinter && s1 == s2
                #(subrows == subcols && rowoffsetblock == coloffsetblock) || throw(DimensionMismatch("bug in hamiltonian routines! $rowoffsetblock, $coloffsetblock, $subrows, $subcols"))
                appendonsites!(I, J, V, rowoffsetblock, lat.sublats[s1], onsite(model, s1), Val(subrows))
            end
            if isvalidlink(isinter, (s2, s1))
                appendhoppings!(I, J, V, (rowoffsetblock, coloffsetblock), ilink.slinks[s2, s1], hopping(model, (s2, s1)), Val(subrows), Val(subcols), !isinter)
            end
            rowoffsetblock += subrows * nsites(lat.sublats[s2])
        end
        coloffsetblock += subcols * nsites(lat.sublats[s1])
    end
    return nothing
end

function appendonsites!(I, J, V, offsetblock, sublat, ons, ::Val{N}) where N
    offset = offsetblock
    for r in sublat.sites
        o = ons(r, Val(N))
        for inds in CartesianIndices(o)
            append!(I, offset + inds[1])
            append!(J, offset + inds[2])
        end
        append!(V, o)
        offset += N
    end
    return nothing
end

function appendhoppings!(I, J, V, (rowoffsetblock, coloffsetblock), slink, hop, ::Val{M}, ::Val{N}, symmetrize) where {M,N}
    posstart = length(I)
    for src in sources(slink), (target, rdr) in neighbors_rdr(slink, src)
        rowoffset = (target - 1) * M
        coloffset = (src - 1) * N
        h = hop(rdr, Val(M), Val(N))
        for inds in CartesianIndices(h)
            append!(I, rowoffsetblock + rowoffset + inds[1])
            append!(J, coloffsetblock + coloffset + inds[2])
        end
        append!(V, h)
    end
    posend = length(I)
    if symmetrize
        # We assume only uniquelinks in intralinks. We add the hermitian conjugate part
        # This should be removed if isvalidlink does not filter out half of the links
        append!(I, view(J, (posstart+1):posend))
        append!(J, view(I, (posstart+1):posend))
        sizehint!(V, posend + (posend - posstart))
        for k in (posstart+1):posend
            @inbounds push!(V, conj(V[k]))
        end
    end
    return nothing
end