#######################################################################
# State
#######################################################################
struct State{L,O,V,T,LP,A<:AbstractArray{V,LP},D<:Supercell{L,O,LP}}
    vector::A
    phases::SVector{O,T}
    supercell::D
end

State(lat::Lattice{E,L,T};
      type::Type{Tv} = Complex{T},
      vector = OffsetArray{orbitaltype(lat, type)}(undef, axes(lat.supercell.cellmask)),
      phases = zerophases(lat)) where {E,L,T,Tv} =
    State(vector, phases, lat.supercell)

zerophases(lat::Lattice{E,L,T}) where {E,L,T} =
    zero(SVector{dim(lat.supercell),T})

Base.show(io::IO, s::State{L,O,V}) where {L,O,N,Tv,V<:SVector{N,Tv}} = print(io,
"State{$L,$O} : state of an $(L)D lattice with an $(O)D supercell
  Element type     : $Tv
  Max orbital size : $N
  Orbitals        : $(sum(s.supercell.cellmask))")

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
    cellmask = lat.supercell.cellmask
    masksize = size(cellmask)
    norbs = norbitals.(lat.sublats)
    v = rand(T, n * N, masksize...) # for performance, use n×N Floats to build an S
    @inbounds for c in CartesianIndices(masksize)
        site = first(Tuple(c))
        insupercell = cellmask.parent[c]
        norb = norbs[sublat(lat, site)] * insupercell
        for j in 1:N, i in 1:n
            v[i + (j-1)*n, Tuple(c)...] =
                (v[i + (j-1)*n, Tuple(c)...] - T(0.5)) * (j <= norb)
        end
    end
    normalize!(vec(v)) # rmul!(v, 1/sqrt(sum(abs2, v))) also works, slightly faster
    rv = reinterpret(V, v)
    sv = OffsetArray([rv[1, c] for c in CartesianIndices(masksize)], axes(cellmask))
    return State(sv, zerophases(lat), lat.supercell)
end

#######################################################################
# mul!
#######################################################################
function mul!(t::S, ham::Hamiltonian{L}, s::S, α::Number = true, β::Number = false) where {L,O,V,S<:State{L,O,V}}
    C = t.vector
    B = s.vector
    celliter = CartesianIndices(tail(axes(B)))
    cols = 1:size(first(ham.harmonics).h, 2)
    bbox = boundingbox(s.supercell)
    Ninv = pinverse(s.supercell.matrix)
    zeroV = zero(V)
    # Scale target by β
    if β != 1
        β != 0 ? rmul!(C, β) : fill!(C, zeroV)
    end
    # Add α * blochphase * h * source to target
    @inbounds Threads.@threads for ic in celliter
        i = Tuple(ic)
        for h in ham.harmonics
            dn = Tuple(h.dn)
            j, dN = wrap(i .+ dn, bbox)
            α´ = α * cis(s.phases' * Ninv * dN)
            nzv = nonzeros(h.h)
            rv = rowvals(h.h)
            for col in cols
                αxj = B[col, i...] * α´
                for p in nzrange(h.h, col)
                    C[rv[p], j...] += nzv[p] * αxj
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