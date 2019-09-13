#######################################################################
# State
#######################################################################
struct State{L,O,D,V}
    wavefunction::Array{V,D}
    blochphases::SVector{O,Float64}
    domain::Domain{L,O,D}
end

function State(lat::Lattice{E,L,T}; type::Type{Tv} = Complex{T}) where {E,L,T,Tv}
    S = orbitaltype(lat, type)
    wavefunction = Array{S}(undef, size(lat.domain.bitmask))
    blochphases = SVector(0.0 .* lat.domain.openboundaries)
    domain = lat.domain
    return State(wavefunction, blochphases, domain)
end

Base.show(io::IO, s::State{L,O,D,V}) where {L,O,D,N,Tv,V<:SVector{N,Tv}} = print(io,
"State{$L,$O} : state of an $(L)D lattice in an $(O)D domain
  Element type     : $Tv
  Max orbital size : $N")