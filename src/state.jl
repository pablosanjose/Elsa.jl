#######################################################################
# States can be a simple Vector (for <:Lattice) or a SupercellState
#######################################################################
struct SupercellState{V,S<:Supercell,A<:OffsetArray{V}} <: AbstractVector{V}
    vector::A
    supercell::S
end

SupercellState(lat::Superlattice{E,L,T};
      type::Type{Tv} = Complex{T},
      vector = OffsetArray{orbitaltype(lat, Tv)}(undef, maskranges(lat))) where {E,L,T,Tv} =
    SupercellState(vector, lat.supercell)

maskranges(lat::Superlattice{E,L}) where {E,L} = (1:nsites(lat), lat.supercell.cells.indices...)
maskranges(lat::Lattice{E,L}) where {E,L} = (1:nsites(lat),)

displayeltype(s::SupercellState{V}) where {V<:Number} = V
displayeltype(s::SupercellState{V}) where {T,N,V<:SVector{N,T}} = T

displayorbsize(s::SupercellState{V}) where {V<:Number} = 1
displayorbsize(s::SupercellState{V}) where {T,N,V<:SVector{N,T}} = N

Base.show(io::IO, s::SupercellState) = show(io, MIME("text/plain"), s)
Base.show(io::IO, ::MIME"text/plain", s::SupercellState{V,S}) where {V,L,L´,S<:Supercell{L,L´}} =
    print(io,
"SupercellState{$L} : state of an $(L)D Superlattice
  Element type     : $(displayeltype(s))
  Max orbital size : $(displayorbsize(s))
  Sites            : $(nsites(s.supercell))")

function floatsorbs(V::Type{<:SVector}, T)
    n, r = divrem(sizeof(eltype(V)), sizeof(T))
    N = length(V) # orbitals
    r == 0 || throw(
        error("Unexpected error: cannot reinterpret orbital type $V as a number of floats"))
    return n, N
end

function floatsorbs(V::Type{<:Number}, T)
    n, r = divrem(sizeof(V), sizeof(T))
    N = 1 # orbitals
    r == 0 || throw(
        error("Unexpected error: cannot reinterpret orbital type $V as a number of floats"))
    return n, N
end

function maybe_wrapstate(sv, lat::Superlattice)
    o = OffsetArray(sv, maskranges(lat))
    return SupercellState(o, lat.supercell)
end
maybe_wrapstate(sv, lat::Lattice) = sv

# External API #

# build a random state, with zeroed-out padding orbitals, throughout the supercell/unitcell
function randomstate(h::Hamiltonian{LA}; type::Type{Tv} = Complex{T}) where {E,L,T,Tv,LA<:AbstractLattice{E,L,T}}
    lat = h.lattice
    masked = ismasked(lat)
    V = orbitaltype(h.orbitals, type)
    n, N = floatsorbs(V, T)
    masksize = length.(maskranges(lat))
    norbs = length.(h.orbitals)
    v = rand(T, n * N, masksize...) # for performance, use n×N Floats to build an S
    if !all(x -> x == norbs[1], norbs)  # zero out missing orbitals
        @inbounds for c in CartesianIndices(masksize)
            site = first(Tuple(c))
            insupercell = !masked || lat.supercell.mask.parent[c]
            norb = norbs[sublat(lat, site)] * insupercell
            for j in 1:N, i in 1:n
                v[i + (j-1)*n, Tuple(c)...] =
                    (v[i + (j-1)*n, Tuple(c)...] - T(0.5)) * (j <= norb)
            end
        end
    end
    rmul!(v, inv(norm(v))) # normalize! without needing to cast v as vector
    rv = reinterpret(V, v)
    sv = V[rv[1, c] for c in CartesianIndices(masksize)]
    return maybe_wrapstate(sv, lat)
end

Base.copy!(t::S, s::S) where {S<:SupercellState} = SupercellState(copy!(t.vector, s.vector), s.supercell)
Base.copy(s::SupercellState) = SupercellState(copy(s.vector), s.supercell)
Base.similar(s::SupercellState) = SupercellState(similar(s.vector),  s.supercell)
Base.size(s::SupercellState, i...) = size(s.vector, i...)
Base.length(s::SupercellState) = length(s.vector)
Base.getindex(s::SupercellState, i...) = getindex(s.vector, i...)

######################################################################
# mul!
######################################################################
function SparseArrays.mul!(t::S, hb::SupercellBloch, s::S, α::Number = true, β::Number = false) where {V,S<:SupercellState{V}}
    C = t.vector
    B = s.vector
    ham = hb.hamiltonian
    phases = hb.phases
    cells = s.supercell.cells
    cols = 1:size(first(ham.harmonics).h, 2)
    pinvint = pinvmultiple(s.supercell.matrix)
    zeroV = zero(V)
    # Scale target by β
    if β != 1
        β != 0 ? rmul!(C, β) : fill!(C, zeroV)
    end
    # Add α * blochphase * h * source to target
    Threads.@threads for ic in cells
        i = Tuple(ic)
        # isemptycell(s, i) && continue # good for performance? No much
        for h in ham.harmonics
            olddn = h.dn + SVector(i)
            newdn = new_dn(olddn, pinvint)
            j = Tuple(wrap_dn(olddn, newdn, s.supercell.matrix))
            CartesianIndex(j) in cells || continue # boundaries in unwrapped directions
            α´ = α * cis(phases' * newdn)
            nzv = nonzeros(h.h)
            rv = rowvals(h.h)
            for col in cols
                @inbounds αxj = B[col, i...] * α´
                for p in nzrange(h.h, col)
                    @inbounds C[rv[p], j...] += applyfield(ham.field, nzv[p], rv[p], col, h.dn) * αxj
                end
            end
        end
    end
    # Filter out sites not in supercell
    @simd for j in eachindex(t.vector)
        @inbounds isinmask(s.supercell, j) || (t.vector[j] = zeroV)
    end
    return t
end

# function isemptycell(s::SupercellState, cell)
#     ismasked(s.supercell) || return false
#     @inbounds for i in size(s.supercell.mask, 1)
#         s.supercell.mask[i, cell...] && return false
#     end
#     return true
# end