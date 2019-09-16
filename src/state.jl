#######################################################################
# State
#######################################################################
struct State{L,O,V,T,A<:AbstractArray{<:AbstractVector{V},L}}
    vector::A
    phases::SVector{O,T}
    domain::Domain{L,O}
end

State(lat::Lattice{E,L,T};
      type::Type{Tv} = Complex{T},
      vector = Array{Vector{orbitaltype(lat, type)}}(undef, size(lat.domain.cellmask)),
      phases = zerophases(lat)) where {E,L,T,Tv} =
    State(vector, phases, lat.domain)

zerophases(lat::Lattice{E,L,T}) where {E,L,T} =
    zero(SVector{length(lat.domain.openboundaries),T})

Base.show(io::IO, s::State{L,O,V}) where {L,O,N,Tv,V<:SVector{N,Tv}} = print(io,
"State{$L,$O} : state of an $(L)D lattice in an $(O)D domain
  Element type     : $Tv
  Max orbital size : $N
  Orbtitals        : $(sum(length, s.vector))")

Base.copy!(t::S, s::S) where {S<:State} = State(copy!(t.vector, s.vector), s.phases, s.domain)
Base.similar(s::State) = State(similar.(s.vector), s.phases, s.domain)
Base.copy(s::State) = State(copy(s.vector), s.phases, s.domain)

# API #

function randomstate(lat::Lattice{E,L,T}, type::Type{Tv} = Complex{T}) where {E,L,T,Tv}
    V = orbitaltype(lat, type)
    n, r = divrem(sizeof(eltype(V)), sizeof(T))
    N = length(V)
    r == 0 || throw(error("Unexpected error: cannot reinterpret orbital type $V as a number of floats"))
    cellmask = lat.domain.cellmask
    nsites = length(first(cellmask))
    norbs = norbitals.(lat.sublats)
    v = rand(T, n * N, nsites, size(cellmask)...) # for performance, use n×N Floats to build an S
    for c in CartesianIndices(cellmask)
        @inbounds for (site, indomain) in enumerate(cellmask[c])
            norb = norbs[sublat(lat, site)] * indomain
            for j in 1:N, i in 1:n
                v[i + (j-1)*n, site, Tuple(c)...] =
                    (v[i + (j-1)*n, site, Tuple(c)...] - T(0.5)) * (j <= norb)
            end
        end
    end
    normalize!(vec(v)) # rmul!(v, 1/sqrt(sum(abs2, v))) also works, slightly faster
    rv = reinterpret(V, v)
    sv = [rv[1, :, c] for c in CartesianIndices(cellmask)]
    return State(sv, zerophases(lat), lat.domain)
end

#######################################################################
# mul!
#######################################################################
function mul!(t::S, h::HamiltonianHarmonic{L}, s::S, α::Number, β::Number) where {L,O,V,S<:State{L,O,V}}
    dn = CartesianIndex(Tuple(h.dn))
    c0 = CartesianIndices(s.domain.cellmask)
    cdn = c0 .- dn
    maxc = min(CartesianIndex(maximum.(c0.indices)), CartesianIndex(maximum.(cdn.indices)))
    minc = max(CartesianIndex(minimum.(c0.indices)), CartesianIndex(minimum.(cdn.indices)))
    @inbounds for i in minc:maxc
        if iszero(s.domain.cellmask[i + dn])
            t.vector[i + dn] .= zero(V)
        else
            mul!(t.vector[i + dn], h.h, s.vector[i], α, β)
        end
    end
    return t
end

function mul!(t::S, ham::Hamiltonian{L}, s::S, α::Number = true, β::Number = false) where {L,O,S<:State{L,O}}
    β´ = β
    for h in ham.harmonics
        mul!(t, h, s, α, β´)
        β´ = one(β)
    end
    return t
end
