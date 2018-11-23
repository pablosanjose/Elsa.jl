#######################################################################
# System
#######################################################################

struct System{T,E,L,M<:Model,EL}
	lattice::Lattice{T,E,L,EL}
    model::M
    hbloch::BlochOperator{T,L}
	vbloch::BlochVector{T,L}
end

function System(l::Lattice, m::Model)
	hop = hamiltonianoperator(l, m)
	vop = velocityoperator(hop)
	return System(l, m, hop, vop)
end

#######################################################################
# Display
#######################################################################

function Base.show(io::IO, sys::System{T,E,L}) where {T,E,L}
    print(io, "System{$T,$E,$L} : $(L)D system in $(E)D space with $T sites.
    Bravais vectors : $(vectorsastuples(sys.lattice))
    Number of sites : $(nsites(sys.lattice))
    Sublattice names : $((sublatnames(sys.lattice)... ,))
    Unique Links : $(nlinks(sys.lattice))
    Model with sublattice site dimensions $((sys.model.dims...,)) (default $(sys.model.defdim))
    $(sys.hbloch)")
end

#######################################################################
# build hamiltonian
#######################################################################

velocityoperator(hbloch::BlochOperator{T,L}; kn = zero(SVector{L,Int}), axis = 1) where {T,L} =
	gradient(hbloch; kn = kn, axis = axis)
function hamiltonianoperator(lat::Lattice{T,E,L}, model::Model) where {T,E,L}
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

    return BlochOperator(I, J, V, Voffsets, Vn, ndist, workspace, mat)
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

updateoperatormatrix!(op) = sparse!(op.matrix, op.I, op.J, op.V, size(op.matrix, 1), op.workspace)

function hbloch!(I, J, V, model, lat, ilink, isinter)
    sdims = sublatdims(lat, model) # orbitals per site of each sublattice
    coloffsetblock = 0
    for (s1, subcols) in enumerate(sdims)
        rowoffsetblock = 0
        for (s2, subrows) in enumerate(sdims)
            if !isinter && s1 == s2
                appendonsites!(I, J, V, rowoffsetblock, lat.sublats[s1], onsite(model, s1), Val(subrows))
            end
            if isvalidlink(isinter, (s1, s2))
                appendhoppings!(I, J, V, (rowoffsetblock, coloffsetblock), ilink.slinks[s2, s1], hopping(model, (s2, s1)), Val(subrows), Val(subcols), !isinter)
            end
            rowoffsetblock += subrows * nsites(lat.sublats[s2])
        end
        coloffsetblock += subcols * nsites(lat.sublats[s1])
    end
    return nothing
end

appendonsites!(I, J, V, offsetblock, sublat, ons::NoOnsite, ::Val) = nothing
function appendonsites!(I, J, V, offsetblock, sublat, ons, ::Val{subrows}) where {subrows}
    offset = offsetblock
    for r in sublat.sites
        o = ons(r, Val(subrows))
        for inds in CartesianIndices(o)
            append!(I, offset + inds[1])
            append!(J, offset + inds[2])
        end
        append!(V, real(o))
        offset += subrows
    end
    return nothing
end

appendhoppings!(I, J, V, (rowoffsetblock, coloffsetblock), slink, hop::NoHopping, ::Val, ::Val, symmetrize) = nothing
function appendhoppings!(I, J, V, (rowoffsetblock, coloffsetblock), slink, hop, ::Val{subrows}, ::Val{subcols}, symmetrize) where {subrows, subcols}
    posstart = length(I)
    for src in sources(slink), (target, rdr) in neighbors_rdr(slink, src)
        rowoffset = (target - 1) * subrows
        coloffset = (src - 1) * subcols
        h = hop(rdr, Val(subrows), Val(subcols))
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

function blochphases(k, sys::System{T,E}) where {T,E}
	length(k) == E || throw(DimensionMismatch("The dimension of the Bloch vector `k` should math the embedding dimension $E"))
	return transpose(bravaismatrix(sys.lattice)) * SVector(k) / (2pi)
end

function hamiltonian(sys::System{T,E,L}; k = zero(SVector{E,T}), kn = blochphases(k, sys), intracell::Bool = false) where {T,E,L}
	length(kn) == L || throw(DimensionMismatch("The dimension of the normalized Bloch phases `kn` should match the lattice dimension $L"))
	nsertblochphases!(sys.hbloch, kn, intracell)
    updateoperatormatrix!(sys.hbloch)
    return sys.hbloch.matrix
end

function velocity(sys::System{T,E,L}; k = zero(SVector{E,T}), kn = blochphases(k, sys), axis::Int = 1) where {T,E,L}
	0 <= axis <= max(L, 1) || throw(DimensionMismatch("Keyword `axis` should be between 0 and $L, the lattice dimension"))
	length(kn) == L || throw(DimensionMismatch("The dimension of the normalized Bloch phases `kn` should match the lattice dimension $L"))
	insertblochphases!(sys.vbloch, kn, axis)
	updateoperatormatrix!(sys.vbloch)
	return sys.vbloch.matrix
end
