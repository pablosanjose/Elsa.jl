#######################################################################
# HermitianOperator
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

struct HermitianOperator{T,L}
    I::Vector{Int}
    J::Vector{Int}
    V::Vector{Complex{T}}
    Voffsets::Vector{Int}
    Vn::Vector{Vector{Complex{T}}}
    ndist::Vector{SVector{L, Int}}
    workspace::SparseWorkspace{T}
    matrix::SparseMatrixCSC{Complex{T},Int}
end

function Base.show(io::IO, op::HermitianOperator{T,L}) where {T,L}
    print(io, "Hermitian operator of size $(size(op.matrix, 1)) with $(nnz(op.matrix)) elements")
end

#######################################################################
# build hamiltonian
#######################################################################

function hamiltonian(lat::Lattice{T,E,L}, model::Model) where {T,E,L}
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

    return HermitianOperator(I, J, V, Voffsets, Vn, ndist, workspace, mat)
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

updatehamiltonian!(h::HermitianOperator) = sparse!(h.matrix, h.I, h.J, h.V, size(h.matrix, 1), h.workspace)

function hbloch!(I, J, V, model, lat, ilink, isinter)
    sdims = sublatdims(lat, model) # orbitals per site of each sublattice
    coloffsetblock = 0
    for (s1, subcols) in enumerate(sdims)
        rowoffsetblock = 0
        for (s2, subrows) in enumerate(sdims)
            if !isinter && s1 == s2
                appendonsites!(I, J, V, rowoffsetblock, lat.sublats[s1], onsite(model, s1), subrows)
            end
            if isvalidlink(isinter, (s1, s2))
                appendhoppings!(I, J, V, (rowoffsetblock, coloffsetblock), ilink.slinks[s2, s1], hopping(model, (s2, s1)), subrows, subcols, !isinter)
            end
            rowoffsetblock += subrows * nsites(lat.sublats[s2])
        end
        coloffsetblock += subcols * nsites(lat.sublats[s1])
    end
    return nothing
end

appendonsites!(I, J, V, offsetblock, sublat, ons::NoOnsite, subrows) = nothing
function appendonsites!(I, J, V, offsetblock, sublat, ons, subrows)
    offset = offsetblock
    for r in sublat.sites
        o = ons(r, Val(subrows))
        for inds in CartesianIndices(o)
            append!(I, offset + inds[1])
            append!(J, offset + inds[2])
        end
        append!(V, o)
        offset += N
    end
    return nothing
end

appendhoppings!(I, J, V, (rowoffsetblock, coloffsetblock), slink, hop::NoHopping, subrows, subcols, symmetrize) = nothing
function appendhoppings!(I, J, V, (rowoffsetblock, coloffsetblock), slink, hop, subrows, subcols, symmetrize)
    posstart = length(I)
    for src in sources(slink), (target, rdr) in neighbors_rdr(slink, src)
        rowoffset = (target - 1) * M
        coloffset = (src - 1) * N
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
