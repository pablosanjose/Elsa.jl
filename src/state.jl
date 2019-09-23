#######################################################################
# State
#######################################################################
struct State{L,V,T,S<:Union{Missing,Supercell{L}},A<:OffsetArray{V}}
    vector::A
    phases::SVector{L,T}
    supercell::S
end

State(lat::Lattice{E,L,T};
      type::Type{Tv} = Complex{T},
      vector = OffsetArray{orbitaltype(lat, Tv)}(undef, cellmaskaxes(lat)),
      phases = zerophases(lat)) where {E,L,T,Tv} =
    State(vector, phases, lat.supercell)

# zerophases(lat::Lattice{E,L,T,Missing}) where {E,L,T} = zero(SVector{L,T})
zerophases(lat::Lattice{E,L,T}) where {E,L,T} = zero(SVector{dim(lat.supercell),T})

# cellmaskaxes(lat::Lattice{E,L,T,Missing}) where {E,L,T} = (1:nsites(lat), ntuple(_->0:0, Val(L))...)
cellmaskaxes(lat::Lattice{E,L}) where {E,L} = axes(lat.supercell.cellmask)

# nsites(s::State{L,V,T,Missing}) where {L,V,T} = length(s.vector)
nsites(s::State) = nsites(s.supercell)

# isemptycell(s::State{L,V,T,Missing}, cell) where {L,V,T} = false
function isemptycell(s::State{L,V,T}, cell) where {L,V,T}
    @inbounds for i in size(s.supercell.cellmask, 1)
        s.supercell.cellmask[i, cell...] && return false
    end
    return true
end

# boundingbox(s::State) = extrema.(tail(axes(s.vector)))

Base.show(io::IO, s::State{L,V}) where {L,N,Tv,V<:SVector{N,Tv}} = print(io,
"State{$L} : state of an $(L)D lattice or superlattice
  Element type     : $Tv
  Max orbital size : $N
  Sites            : $(nsites(s))")

Base.copy!(t::S, s::S) where {S<:State} = State(copy!(t.vector, s.vector), s.phases, s.supercell)
Base.copy(s::State) = State(copy(s.vector), s.phases, s.supercell)
Base.similar(s::State) = State(similar(s.vector), s.phases, s.supercell)

# API #

function randomstate(lat::Lattice{E,L,T}, type::Type{Tv} = Complex{T}) where {E,L,T,Tv}
    V = orbitaltype(lat, type)
    n, r = divrem(sizeof(eltype(V)), sizeof(T))
    N = length(V)
    r == 0 || throw(
        error("Unexpected error: cannot reinterpret orbital type $V as a number of floats"))
    masksize = length.(cellmaskaxes(lat))
    norbs = length.(lat.unitcell.orbitals)
    v = rand(T, n * N, masksize...) # for performance, use n×N Floats to build an S
    @inbounds for c in CartesianIndices(masksize)
        site = first(Tuple(c))
        insupercell = lat.supercell.cellmask.parent[c]
        norb = norbs[sublat(lat, site)] * insupercell
        for j in 1:N, i in 1:n
            v[i + (j-1)*n, Tuple(c)...] =
                (v[i + (j-1)*n, Tuple(c)...] - T(0.5)) * (j <= norb)
        end
    end
    rmul!(v, inv(norm(v))) # normalize! without needing to cast v as vector
    rv = reinterpret(V, v)
    sv = OffsetArray([rv[1, c] for c in CartesianIndices(masksize)], cellmaskaxes(lat))
    return State(sv, zerophases(lat), lat.supercell)
end

#######################################################################
# mul!
#######################################################################
function mul!(t::S, ham::Hamiltonian{L}, s::S, α::Number = true, β::Number = false) where {L,V,S<:State{L,V}}
    C = t.vector
    B = s.vector
    celliter = CartesianIndices(tail(axes(B)))
    cols = 1:size(first(ham.harmonics).h, 2)
    pinvint = pinvmultiple(s.supercell.matrix)
    zeroV = zero(V)
    # Scale target by β
    if β != 1
        β != 0 ? rmul!(C, β) : fill!(C, zeroV)
    end
    # Add α * blochphase * h * source to target
    @inbounds Threads.@threads for ic in celliter
        i = Tuple(ic)
        # isemptycell(s, i) && continue # good for performance? Check
        for h in ham.harmonics
            olddn = h.dn + SVector(i)
            newdn = new_dn(olddn, pinvint)
            j = Tuple(wrap_dn(olddn, newdn, s.supercell.matrix))
            α´ = α * cis(s.phases' * newdn)
            nzv = nonzeros(h.h)
            rv = rowvals(h.h)
            for col in cols
                αxj = B[col, i...] * α´
                for p in nzrange(h.h, col)
                    C[rv[p], j...] += applyfield(ham.field, nzv[p], rv[p], col, h.dn) * αxj
                end
            end
        end
    end
    # Filter out sites not in supercell
    @simd for j in eachindex(t.vector)
        @inbounds s.supercell.cellmask[j] || (t.vector[j] = zeroV)
    end
    return t
end