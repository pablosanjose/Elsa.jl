struct BrillouinMesh{T,E,N} # N = E + 1, where E is the dimension of the lattice
    vertices::Vector{SVector{E,T}}
    elements::Vector{SVector{N,Int}}
end

function BrillouinMesh(lat::Lattice{T,E,L}) where {T,E,L}
    vertices = SVector{E,T}[]
    for s in lat.sublats
        append!(vertices, s.sites)
    end
    nverts = length(vertices)
    
    model = Model(Hopping(1))
    adjacency = hamiltonian(lat, model).matrix
    size(adjacency, 1) == nverts || throw(DimensionMismatch("mesh hamiltonian dimension does not match number of vertices"))
    
    flatgroups = Int[]
    for n in 1:E  # We grow groups recursively till (n+1)-groups
        flatgroups = newgroups(flatgroups, adjacency, n)
    end    

    elements = collect(reinterpret(SVector{E+1,Int}, flatgroups))

    return BrillouinMesh(vertices, elements)
end

# Computes the groups of n+1 vertices such that all members are neighbors of the all the rest
function newgroups(flatgroups, adjacency, n) # n is the number of vertices in each seed group
    if n == 1  # We start with 1-groups, one per vertex
        seedmat = sparse(1I, size(adjacency))
        stepmat = adjacency
        flatgroups = rowvals(seedmat)
        ngroups = length(flatgroups)
    else
        nvertices = size(adjacency, 1)
        ngroups::Int = length(flatgroups) / n
        colptr = collect(1:n:(length(flatgroups) + 1))
        nzval = fill(1, length(flatgroups))
        seedmat = SparseMatrixCSC(nvertices, ngroups, colptr, flatgroups, nzval)
        stepmat = adjacency * seedmat
    end

    steprows = rowvals(stepmat)
    stepvals = nonzeros(stepmat)
    prevgroup = Vector{Int}(undef, n)
    newflatgroups = Int[]

    for col in 1:ngroups
        ind = 1  # copy to prevgroup the seed column indices, to be expanded
        for i in nzrange(seedmat, col)
            prevgroup[ind] = flatgroups[i]
            ind += 1
        end
        lastseed = prevgroup[end]
        for j in nzrange(stepmat, col)
            if steprows[j] > lastseed && stepvals[j] == n 
                append!(newflatgroups, prevgroup)
                push!(newflatgroups, steprows[j])
            end
        end
    end
    
    return newflatgroups
end

#######################################################################
# BrillouinMesh display
#######################################################################

Base.show(io::IO, mesh::BrillouinMesh{T,E,N}) where {T,E,N} =
    print(io, "BrillouinMesh{$T,$E,$N} : $(E)D mesh with $(length(mesh.vertices)) vertices and $(length(mesh.elements)) elements ($N vertices each)")