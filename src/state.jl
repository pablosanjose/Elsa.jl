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
      vector = Array{Vector{orbitaltype(lat, type)}}(undef, size(lat.domain.bitmask)),
      phases = zerophases(lat)) where {E,L,T,Tv} =
    State(vector, phases, lat.domain)

zerophases(lat::Lattice{E,L,T}) where {E,L,T} =
    zero(SVector{length(lat.domain.openboundaries),T})

Base.show(io::IO, s::State{L,O,V}) where {L,O,N,Tv,V<:SVector{N,Tv}} = print(io,
"State{$L,$O} : state of an $(L)D lattice in an $(O)D domain
  Element type     : $Tv
  Max orbital size : $N
  Orbtitals        : $(length(s.vector))")

Base.copy!(t::S, s::S) where {S<:State} = State(copy!(t.vector, s.vector), s.phases, s.domain)

# API #

function randomstate(lat::Lattice{E,L,T}, type::Type{Tv} = Complex{T}) where {E,L,T,Tv}
    V = orbitaltype(lat, type)
    n, r = divrem(sizeof(eltype(V)), sizeof(T))
    N = length(V)
    r == 0 || throw(error("Unexpected error: cannot reinterpret orbital type $V as a number of floats"))
    bitmask = lat.domain.bitmask
    nsites = length(first(bitmask))
    norbs = norbitals.(lat.sublats)
    v = rand(T, n, N, nsites, size(bitmask)...) # for performance, use n×N Floats to build an S
    @inbounds for c in CartesianIndices(bitmask)
        for (site, indomain) in enumerate(bitmask[c])
            norb = norbs[sublat(lat, site)] * indomain
            for j in 1:N, i in 1:n
                v[i, j, site, Tuple(c)...] = (v[i, j, site, Tuple(c)...] - T(0.5)) * (j <= norb)
            end
        end
    end
    v0 = vec(v)
    normalize!(v0) # an unsafe, faster rmul!(v0, 1/sqrt(sum(abs2, v0))) works too (v0~1)
    rv = reshape(reinterpret(V, v0), nsites, size(bitmask)...)
    sv = [rv[:, c] for c in CartesianIndices(bitmask)]
    return State(sv, zerophases(lat), lat.domain)
end

#######################################################################
# mul!
#######################################################################
function LinearAlgebra.mul!(t::S, h::HamiltonianHarmonic{L}, s::S,
                            α::Number, β::Number) where {L,O,S<:State{L,O}}
    shift = (0, h.dn...)
end