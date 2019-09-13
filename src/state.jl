#######################################################################
# State
#######################################################################
struct State{D,O,V,T,A<:AbstractArray{V,D}}
    vector::A
    phases::SVector{O,T}
end

State(lat::Lattice{E,L,T};
      type::Type{Tv} = Complex{T},
      vector = Array{orbitaltype(lat, type)}(undef, size(lat.domain.bitmask)),
      phases = zerophases(lat)) where {E,L,T,Tv} =
    State(vector, phases)

zerophases(lat::Lattice{E,L,T}) where {E,L,T} =
    zero(SVector{length(lat.domain.openboundaries),T})

Base.show(io::IO, s::State{D,O,V}) where {D,O,N,Tv,V<:SVector{N,Tv}} = print(io,
"State{$(D-1),$O} : state of an $(D-1)D lattice in an $(O)D domain
  Element type     : $Tv
  Max orbital size : $N
  Orbtitals        : $(length(s.vector))")

function randomstate(lat::Lattice{E,L,T}, type::Type{<:Number} = Complex{T}) where {E,L,T,Tv}
    S = orbitaltype(lat, type)
    N = length(S)
    n, r = divrem(sizeof(eltype(S)), sizeof(Float64))
    r == 0 || throw(error("Unexpected error: cannot reinterpret orbital type $S as a number of floats"))
    bitmask = lat.domain.bitmask
    norbs = norbitals.(lat.sublats)
    v = rand(n, N, length(bitmask)) # for performance, use n×N Floats to build an S
    for (k, indomain) in enumerate(bitmask)
        s = sublat(lat, k)
        @inbounds for j in 1:N, i in 1:n
            v[i,j,k] *= indomain && j <= norbs[s]
        end
    end
    v0 = vec(v)
    normalize!(v0) # an unsafe, faster rmul!(v0, 1/sqrt(sum(abs2, v0))) works too (v0~1)
    rv = reinterpret(S, v0)
    return State(rv, zerophases(lat))
end

mask(S::Type{V}, orbs) where {N,T,V<:SVector{N,T}} =
    SVector{N,Int}(ntuple(i -> i <= orbs ? 1 : 0, Val(N)))