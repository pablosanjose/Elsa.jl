#######################################################################
# Elements
#######################################################################

struct Elements{N}
    indices::Vector{SVector{N,Int}}
end

function Elements(lat::Lattice{T,E}, ::Val{N} = Val(E+1); sublat::Int = 1) where {T,E,N} 
    inds = elements(lat.links.intralink.slinks[sublat,sublat], Val(N))
    alignnormals!(inds, lat.sublats[sublat].sites)
    return Elements(inds)
end
        
Base.show(io::IO, elements::Elements{N}) where {N} = 
    print(io, "Elements{$N}: $(nelements(elements)) elements ($N-vertex)")

nelements(el::Elements) = length(el.indices)

function elements(slink::Slink{T,E}, ::Val{N}) where {T,E,N} 
    indices = SVector{N,Int}[]
    isempty(slink) && return indices
    candidates = SVector{N,Int}[]
    buffer1 = Int[]
    buffer2 = Int[]
    for src in sources(slink)
        resize!(candidates, 0)
        push!(candidates, modifyat(zero(SVector{N,Int}), 1, src))
        imax = 0
        for pass in 2:N
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

alignnormals!(elements, sites) = elements
function alignnormals!(elements::Vector{SVector{3,Int}}, sites::Vector{SVector{E,T}}) where {E,T}
    for (i, element) in enumerate(elements)
        s1 = padright(sites[element[2]] - sites[element[1]], zero(T), Val(3)) 
        s2 = padright(sites[element[3]] - sites[element[1]], zero(T), Val(3)) 
        cross(s1, s2)[3] < 0 && (elements[i] = reverse(element))
    end
    return elements
end

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
    mesh::Lattice{T,L,L,LL}
    uniform::Bool
    partitions::NTuple{L,Int}
    elements::Elements{N}
end

function BrillouinMesh(lat::Lattice{T,E,L}; uniform::Bool = false, partitions = 5) where {T,E,L}
    partitions_tuple = tontuple(Val(L), partitions)
    if uniform
        mesh = uniform_mesh(lat, partitions_tuple)
    else
        mesh = simple_mesh(lat, partitions_tuple)
    end
    # wrappedmesh = wrap(mesh)
    elements = Elements(mesh)
    return BrillouinMesh(mesh, uniform, partitions_tuple, elements)
end
BrillouinMesh(sys::System; kw...) = BrillouinMesh(sys.lattice; kw...)

Base.show(io::IO, m::BrillouinMesh{T,L,N}) where {T,L,N} =
    print(io, "BrillouinMesh{$T,$L} : discretization of $L-dimensional Brillouin zone
    Mesh type  : $(m.uniform ? "uniform" : "simple")
    Vertices   : $(nsites(m.mesh)) 
    Partitions : $(m.partitions)
    $N-elements : $(nelements(m))")

nelements(m::BrillouinMesh) = nelements(m.elements)

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