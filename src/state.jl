#######################################################################
# SupercellState
#######################################################################
struct SupercellBloch{L,T,M,H<:Hamiltonian{<:Superlattice,L,M}}
    hamiltonian::H
    phases::SVector{L,T}
end

Base.summary(h::SupercellBloch{L,T,M}) where {L,T,M} =
    "SupercellBloch{$L,$(eltype(M))}: Bloch Hamiltonian matrix lazily defined on a supercell"

function Base.show(io::IO, sb::SupercellBloch)
    ioindent = IOContext(io, :indent => string("  "))
    print(io, summary(sb), "\n  Phases          : $(Tuple(sb.phases))\n")
    print(ioindent, sb.hamiltonian.lattice.supercell)
end

#######################################################################
# SupercellState
#######################################################################
struct SupercellState{V,S<:Union{Missing,Supercell},A<:OffsetArray{V}}
    vector::A
    supercell::S
end

SupercellState(lat::Superlattice{E,L,T};
      type::Type{Tv} = Complex{T},
      vector = OffsetArray{orbitaltype(lat, Tv)}(undef, cellmaskaxes(lat))) where {E,L,T,Tv} =
    SupercellState(vector, lat.supercell)

cellmaskaxes(lat::Superlattice{E,L}) where {E,L} = axes(lat.supercell.mask)
cellmaskaxes(lat::Lattice{E,L}) where {E,L} = (1:nsites(lat),)

nsites(s::SupercellState) = nsites(s.supercell)

function isemptycell(s::SupercellState, cell)
    @inbounds for i in size(s.supercell.mask, 1)
        s.supercell.mask[i, cell...] && return false
    end
    return true
end

Base.show(io::IO, s::SupercellState{V,S}) where {N,Tv,V<:SVector{N,Tv},L,L´,S<:Supercell{L,L´}} =
    print(io,
"SupercellState{$L} : state of an $(L)D Superlattice
  Element type     : $Tv
  Max orbital size : $N
  Sites            : $(nsites(s))")

Base.copy!(t::S, s::S) where {S<:SupercellState} = SupercellState(copy!(t.vector, s.vector), s.supercell)
Base.copy(s::SupercellState) = SupercellState(copy(s.vector), s.supercell)
Base.similar(s::SupercellState) = SupercellState(similar(s.vector),  s.supercell)

# External API #

# build a random state, with zeroed-out padding orbitals, throughout the supercell/unitcell
function randomstate(lat::AbstractLattice{E,L,T}; type::Type{Tv} = Complex{T}) where {E,L,T,Tv}
    isslat = issuperlattice(lat)
    V = orbitaltype(lat, type)
    n, N = floatsorbs(V, T)
    masksize = length.(cellmaskaxes(lat))
    norbs = length.(lat.unitcell.orbitals)
    v = rand(T, n * N, masksize...) # for performance, use n×N Floats to build an S
    if !all(x -> x == norbs[1], norbs)  # zero out missing orbitals
        @inbounds for c in CartesianIndices(masksize)
            site = first(Tuple(c))
            insupercell = !isslat || lat.supercell.mask.parent[c]
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

# # Auxiliary #

# function randomstate2(lat::Lattice{E,L,T}; type::Type{Tv} = Complex{T}) where {E,L,T,Tv}
#     v = maskedrand(lat, type, 1:nsites(lat))
#     return v
# end

# function randomstate2(lat::Superlattice{E,L,T}; type::Type{Tv} = Complex{T}) where {E,L,T,Tv}
#     v = maskedrand(lat, type, cellmaskaxes(lat)...)
#     return SupercellState(v, lat.supercell)
# end

# function maskedrand(lat, type, siteiter, celliter...)
#     V = orbitaltype(lat, type)
#     lcell = length.(celliter)
#     lsite = length(siteiter)
#     v = rand(V, lsite, lcell...)
#     norbs = length.(lat.unitcell.orbitals)
#     for c in CartesianIndices(lcell), s in 1:nsublats(lat)
#         norb = norbs[s]
#         for j in siterange(lat, s)
#             v[j, Tuple(c)...]
#             # v[j, Tuple(c)...] = mask(v[j, Tuple(c)...], norb)
#             # v[j, Tuple(c)...] *= mask(V, norb)
#         end
#     end
#     # rmul!(v, inv(norm(v)))
#     # return OffsetArray(v, siteiter, celliter...)
# end

# mask(::Type{SVector{L,T}}, norb) where {L,T} =
#     SMatrix{L,L,T}(Diagonal(SVector(ntuple(i -> i > norb ? zero(T) : one(T), Val(L)))))
# mask(::Type{T}, norb) where {T<:Number} = one(T)
# mask(s::SVector{L,T}, norb) where {L,T} =
#     SVector(ntuple(i -> i > norb ? zero(T) : s[i], Val(L)))
# mask(s::Number, norb) = s

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
    o = OffsetArray(sv, cellmaskaxes(lat))
    return SupercellState(o, lat.supercell)
end
maybe_wrapstate(sv, lat::Lattice) = sv

#######################################################################
# mul!
#######################################################################
# function mul!(t::S, ham::Hamiltonian{L}, s::S, α::Number = true, β::Number = false) where {L,V,S<:SupercellState{L,V}}
#     C = t.vector
#     B = s.vector
#     celliter = CartesianIndices(tail(axes(B)))
#     cols = 1:size(first(ham.harmonics).h, 2)
#     pinvint = pinvmultiple(s.supercell.matrix)
#     zeroV = zero(V)
#     # Scale target by β
#     if β != 1
#         β != 0 ? rmul!(C, β) : fill!(C, zeroV)
#     end
#     # Add α * blochphase * h * source to target
#     @inbounds Threads.@threads for ic in celliter
#         i = Tuple(ic)
#         # isemptycell(s, i) && continue # good for performance? Check
#         for h in ham.harmonics
#             olddn = h.dn + SVector(i)
#             newdn = new_dn(olddn, pinvint)
#             j = Tuple(wrap_dn(olddn, newdn, s.supercell.matrix))
#             α´ = α * cis(s.phases' * newdn)
#             nzv = nonzeros(h.h)
#             rv = rowvals(h.h)
#             for col in cols
#                 αxj = B[col, i...] * α´
#                 for p in nzrange(h.h, col)
#                     C[rv[p], j...] += applyfield(ham.field, nzv[p], rv[p], col, h.dn) * αxj
#                 end
#             end
#         end
#     end
#     # Filter out sites not in supercell
#     @simd for j in eachindex(t.vector)
#         @inbounds isinmask(s.supercell, j) || (t.vector[j] = zeroV)
#     end
#     return t
# end