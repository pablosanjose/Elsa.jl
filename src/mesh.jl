#######################################################################
# Elements
#######################################################################

struct Elements{N}
    indices::Vector{SVector{N,Int}}
end

Elements(lat::Lattice, sublat = 1) = Elements(elements(lat.links.intralink.slinks[sublat,sublat]))

Base.show(io::IO, elements::Elements{N}) where {N} = 
    print(io, "Elements{$N}: $(nelements(elements)) elements ($N-vertex)")

nelements(el::Elements) = length(el.indices)

function elements(slink::Slink{T,E}) where {T,E} 
    indices = SVector{E+1,Int}[]
    isempty(slink) && return indices
    candidates = SVector{E+1,Int}[]
    buffer1 = Int[]
    buffer2 = Int[]
    for src in sources(slink)
        resize!(candidates, 0)
        push!(candidates, modifyat(zero(SVector{E+1,Int}), 1, src))
        imax = 0
        for pass in 2:E+1
            (imin, imax) = (imax + 1, length(candidates))
            for i in imin:imax
                neighborbuffer = _common_ordered_neighbors!(buffer1, buffer2, candidates[i], pass - 1, slink)
                for neigh in neighborbuffer
                    push!(candidates, modifyat(candidates[i], pass, neigh))
                end
            end
        end
        (imin, imax) = (imax + 1, length(candidates))
        for i in imin:imax
            push!(indices, candidates[i])
        end
    end
    return indices
end

function _common_ordered_neighbors!(buffer1, buffer2, candidate::SVector{N,Int}, upto, slink) where {N}
    min_neighbor = maximum(candidate)
    resize!(buffer1, 0)
    resize!(buffer2, 0)
    for neigh in neighbors(slink, candidate[1])
        push!(buffer1, neigh)
    end
    for j in 2:upto
        for neigh in neighbors(slink, candidate[j])
            (neigh > min_neighbor) && (neigh in buffer1) && push!(buffer2, neigh)
        end
        buffer1, buffer2 = buffer2, buffer1
    end
    return buffer1
end

#######################################################################
# MeshBrillouin
#######################################################################

struct MeshBrillouin{T,E}
    lattice::Lattice{T,E,0,0}
end

function MeshBrillouin(lat::Lattice{T,E,L}; uniform::Bool = false, partitions = 5) where {T,E,L}
    if uniform
        meshlat = uniform_mesh(lat, partitions)
    else
        meshlat = minimal_mesh(lat, partitions)
    end
    wrappedmesh = wrap(meshlat)
    return MeshBrillouin(wrappedmesh)
end

# phi-space sampling z, k-space G'z. M = diagonal(partitions)
# M G' z =  Tr * n, where n are SVector{L,Int}, and Tr is a hypertriangular lattice
# For some integer S = (n1,n2...), (z1, z2, z3) = I (corners of BZ).
# Hence S = round.(Tr^{-1} G' M) = superlattice. Bravais are z_i for n = I, so simply S^{-1}
# Links should be fixed at the Tr level.
function uniform_mesh(lat::Lattice{T,E,L}, partitions) where {T,E,L}
    M = diagsmatrix(Val(L), partitions)
    A = qr(bravaismatrix(lat)).R
    Gt = qr(transpose(inv(A))).R
    Gt = Gt / Gt[1,1]
    iGt = inv(Gt)
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

hypertriangular(lat::Lattice{T,E,L}) where {T,E,L} = hypertriangular(SMatrix{L,1,T}(I))
function hypertriangular(s::SMatrix{L,L2,T}) where {L,L2,T} 
    v1 = s[:,L2]
    factor = T(1/(L2+1))
    v2 = modifyat(v1, L2, v1[L2]*factor)
    v2 = modifyat(v2, L2 + 1, v1[L2]*sqrt(1 - factor^2))
    return hypertriangular(hcat(s, v2))
end
hypertriangular(s::SMatrix{L,L}) where L = s