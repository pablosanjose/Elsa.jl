#######################################################################
# State
#######################################################################
struct State{L,O,V,T,A<:AbstractArray{<:AbstractVector{V},L},D<:Domain{L,O}}
    vector::A
    phases::SVector{O,T}
    domain::D
end

State(lat::Lattice{E,L,T};
      type::Type{Tv} = Complex{T},
      vector = Array{Vector{orbitaltype(lat, type)}}(undef, size(lat.domain.cellmask)),
      phases = zerophases(lat)) where {E,L,T,Tv} =
    State(vector, phases, lat.domain)

zerophases(lat::Lattice{E,L,T}) where {E,L,T} =
    zero(SVector{nopenboundaries(lat.domain),T})

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
    masksize = size(cellmask)
    nsites = length(first(cellmask))
    norbs = norbitals.(lat.sublats)
    v = rand(T, n * N, nsites, masksize...) # for performance, use n×N Floats to build an S
    for c in CartesianIndices(masksize)
        @inbounds for (site, indomain) in enumerate(cellmask.parent[c])
            norb = norbs[sublat(lat, site)] * indomain
            for j in 1:N, i in 1:n
                v[i + (j-1)*n, site, Tuple(c)...] =
                    (v[i + (j-1)*n, site, Tuple(c)...] - T(0.5)) * (j <= norb)
            end
        end
    end
    normalize!(vec(v)) # rmul!(v, 1/sqrt(sum(abs2, v))) also works, slightly faster
    rv = reinterpret(V, v)
    sv = OffsetArray([rv[1, :, c] for c in CartesianIndices(masksize)], axes(cellmask))
    return State(sv, zerophases(lat), lat.domain)
end

#######################################################################
# mul!
#######################################################################

function mul!(t::S, ham::Hamiltonian{L}, s::S, α::Number = true, β::Number = false) where {L,O,V,S<:State{L,O,V}}
    C = t.vector
    B = s.vector
    cols = 1:size(first(ham.harmonics).h, 2)
    bbox = boundingbox(s.domain)
    Ninv = pinverse(s.domain.openbravais)
    if β != 1
        if β != 0
            Threads.@threads for c in C
                rmul!(c, β)
            end
        else
            Threads.@threads for c in C
                fill!(c, zero(V))
            end
        end
    end
    Threads.@threads for i in CartesianIndices(B)
        bi = B[i]
        for h in ham.harmonics
            dn = CartesianIndex(Tuple(h.dn))
            j, dN = wrap(i + dn, bbox)
            α´ = α * cis(s.phases' * Ninv * dN)
            cj = C[j]
            nzv = nonzeros(h.h)
            rv = rowvals(h.h)
            for col in cols
                αxj = bi[col] * α´
                for p in nzrange(h.h, col)
                    cj[rv[p]] += nzv[p] * αxj
                end
            end
        end
    end
    Threads.@threads for j in eachindex(t.vector)
        t.vector[j] .*= s.domain.cellmask[j]
    end
    return t
end
