#######################################################################
# Bandstructure
#######################################################################
struct Band{M,A<:AbstractVector{M},MD<:Mesh}
    mesh::MD    # Mesh with missing vertices removed
    states::A   # Must be resizeable container to build & refine band
    dimstates::Int
end

function Band{M}(mesh::Mesh, dimstates::Int) where {M}
    nk = nvertices(mesh)
    states = M[]
    return Band(mesh, states, dimstates)
end

struct Bandstructure{D,M,B<:Band{M},MD<:Mesh{D}}   # D is dimension of parameter space
    bands::Vector{B}
    kmesh::MD
end

function Base.show(io::IO, b::Bandstructure{D,M}) where {D,M}
    ioindent = IOContext(io, :indent => string("  "))
    print(io,
"Bandstructure: bands for a $(D)D hamiltonian
  Bands        : $(length(b.bands))
  Element type : $(displayelements(M))")
    print(ioindent, "\n", b.kmesh)
end

#######################################################################
# bandstructure
#######################################################################
 # barrier for type-unstable diagonalizer
function bandstructure(h::Hamiltonian{<:Any,L,M}; resolution = 13, shift = missing, kw...) where {L,M}
    mesh = marchingmesh(h; npoints = resolution, shift = shift)
    return bandstructure!(diagonalizer(h, mesh; kw...), h,  mesh)
end

function bandstructure!(d::Diagonalizer, h::Hamiltonian{<:Lattice,<:Any,M}, mesh::MD) where {M,D,T,MD<:Mesh{D,T}}
    nϵ = d.levels
    dimh = size(h, 1)
    nk = nvertices(mesh)
    ϵks = Matrix{T}(undef, nϵ, nk)
    ψks = Array{M,3}(undef, dimh, nϵ, nk)
    p = Progress(nk, "Step 1/2 - Diagonalising: ")
    for (n, ϕs) in enumerate(vertices(mesh))
        bloch!(d.matrix, h, ϕs)
        (ϵk, ψk) = diagonalize(d)
        resolve_degeneracies!(ϵk, ψk, d, ϕs)
        copyslice!(ϵks, CartesianIndices((1:nϵ, n:n)),
                   ϵk,  CartesianIndices((1:nϵ,)))
        copyslice!(ψks, CartesianIndices((1:dimh, 1:nϵ, n:n)),
                   ψk,  CartesianIndices((1:dimh, 1:nϵ)))
        ProgressMeter.next!(p; showvalues = ())
    end

    bands = Band{M,Vector{M},Mesh{D+1,T,Vector{SVector{D+1,T}},Vector{NTuple{D+1,Int}}}}[]
    bandindices = Vector{Int}(undef, nk) # the 1:nϵ index for each k point. 0 == missing
    p = Progress(nϵ, "Step 2/2 - Connecting bands: ")
    for nb in 1:nϵ
        findbandindices!(bandindices, nb, ψks, mesh, d.minprojection)
        band = extractband(bandindices, ϵks, ψks, mesh)
        push!(bands, band)
        ProgressMeter.next!(p; showvalues = ())
    end
    return Bandstructure(bands, mesh)
end

function findbandindices!(bandindices, nb, ψks, mesh, minprojection)
    dimh, nϵ, nk = size(ψks)
    fill!(bandindices, 0)
    bandindices[1] = nb
    defectfound = false
    for srck in 1:nk, edgek in edges(mesh, srck)
        destk = edgedest(mesh, edgek)
        srcb = bandindices[srck]
        proj, destb = findmostparallel(ψks, destk, srcb, srck)
        if proj > minprojection
            if !iszero(bandindices[destk]) && bandindices[destk] != destb
                defectfound = true
            end
            bandindices[destk] = destb
        end
    end
    defectfound && @warn "Non-trivial band degeneracy detected. Resolution not yet implemented."
    return bandindices
end

function findmostparallel(ψks::Array{M,3}, destk, srcb, srck) where {M}
    T = real(eltype(M))
    dimh, nϵ, nk = size(ψks)
    maxproj = zero(T)
    destb = 0
    srcb == 0 && return maxproj, destb
    @inbounds for nb in 1:nϵ
        proj = zero(M)
        @simd for i in 1:dimh
            proj += ψks[i, nb, destk]' * ψks[i, srcb, srck]
        end
        absproj = T(abs(tr(proj)))
        if maxproj < absproj
            destb = nb
            maxproj = absproj
        end
    end
    return maxproj, destb
end

function extractband(bandindices, ϵks, ψks, mesh::Mesh{D,T}) where {D,T}
    dimh, nϵ, nk = size(ψks)
    states = similar(ψks, dimh * nk)
    vertices = Vector{SVector{D+1,T}}(undef, nk)
    k´ = 0
    for (k, ind) in enumerate(bandindices)
        if !iszero(ind)
            k´ += 1
            vertices[k´] = SVector(Tuple(mesh.vertices[k])..., ϵks[ind, k])
            copyto!(states, 1 + dimh * (k´ - 1), ψks, 1 + dimh * (k - 1), dimh)
            bandindices[k] = k´ # Reuse to store new vertex indices
        end
    end
    if k´ < nk
        resize!(vertices, k´)
        resize!(states, k´ * dimh)
        simplices = extractsimplices(mesh.simplices, bandindices)
        adjmat = extractsadjacencies(mesh.adjmat, bandindices)
    else
        simplices = copy(vec(mesh.simplices))
        adjmat = copy(mesh.adjmat)
    end
    mesh´ = Mesh(vertices, adjmat, simplices)
    band = Band(mesh´, states, dimh)
    return band
end

function extractsimplices(simplices::AbstractVector{NTuple{N,Int}}, indices) where {N}
    simplices´ = similar(simplices)
    n = 0
    for simp in simplices
        simp´ = ntuple(i -> indices[simp[i]], Val(N))
        if all(!iszero, simp´)
            n += 1
            simplices´[n] = simp´
        end
    end
    resize!(simplices´, n)
    return simplices´
end

## This is simpler, but allocates more, and is slower
# extractsadjacencies(adjmat, bandindices) =
#     adjmat[(!iszero).(bandindices), (!iszero).(bandindices)]

function extractsadjacencies(adjmat::AbstractSparseMatrix{Tv}, bandindices) where {Tv}
    n = count(!iszero, bandindices)
    b = SparseMatrixBuilder{Tv}(n, n)
    for col in 1:size(adjmat, 2)
        iszero(bandindices[col]) && continue
        for ptr in nzrange(adjmat, col)
            row = rowvals(adjmat)[ptr]
            iszero(bandindices[row]) || pushtocolumn!(b, row, nonzeros(adjmat)[ptr])
        end
        finalizecolumn!(b)
    end
    adjmat´ = sparse(b)
    return adjmat´
end