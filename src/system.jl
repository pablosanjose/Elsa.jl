#######################################################################
# System
#######################################################################
"""
    System(sublats::Sublat... [, br::Bravais[, model::Model]]; dim::Val, postype, hamtype)

Build a `System{Tv,T,E,L}` of `L` dimensions in `E`-dimensional embedding
space and composed of `T`-typed sites and `Tv`-typed Hamiltonian. See
`Sublat`, `Bravais` and `Model` for syntax.  To indicate a specific embedding 
dimension `E`, use keyword `dim = Val(E)`. Similarly override types `T` and `Tv` 
with `ptype = T` and `htype = Tv`.

    System(presetname::$(NameType)[, model]; kw...)

Build a system from a given preset, and optionally add a `model` to it. Preset 
names are defined in the dictionary `Elsa.systempresets`, together with their 
allowed keyword arguments `kw`. These typically include `norbitals` for sublattices.

    System(sys::System, model::Model)

Build a new system with the same sublattices and bravias vectors as `sys`
but replacing its Hamiltonian with a new `model`.

# Examples
```jldoctest
julia> System(Sublat((1,0), (0,1); name = "C", norbitals = 2), 
              Sublat((0.5,0.5); name = "D"), 
              Bravais((1,0), (0,2)),
              Model(hopping((r,dr) -> @SMatrix[0.1; dr[1]], sublats = (:C, :D), range = 1)))
System{Complex{Float64},Float64,2,2} : 2D system in 2D space
  Bravais vectors     : ((1.0, 0.0), (0.0, 2.0))
  Sublattice names    : (:C, :D)
  Sublattice orbitals : (2, 1)
  Total sites         : 3 [Float64]
  Total hoppings      : 8 [Complex{Float64}]
  Coordination        : 2.6666666666666665

julia> System(:honeycomb, Model(hopping(@SMatrix[1 2; 0 1], sublats = (1,2))), 
              dim = Val(3), htype = Float32, norbitals = 2)
System{Float32,Float64,3,2} : 2D system in 3D space
  Bravais vectors     : ((0.5, 0.866025, 0.0), (-0.5, 0.866025, 0.0))
  Sublattice names    : (:A, :B)
  Sublattice orbitals : (2, 2)
  Total sites         : 2 [Float64]
  Total hoppings      : 6 [Float32]
  Coordination        : 3.0

julia> Tuple(keys(Elsa.systempresets))
(:bcc, :cubic, :honeycomb, :linear, :graphene_bilayer, :square, :triangular)
```
"""
mutable struct System{Tv,T,E,L,EL,S<:Operator{Tv,L}}
    sublats::Vector{Sublat{T,E}}
    sublatsdata::SublatsData
    bravais::Bravais{E,L,EL}
    hamiltonian::S
end

System(s1::Sublat, s2...; kw...) = ((ss, args) = collectfirst(s1, s2...); System([promote(ss...)...], args...; kw...))
System(sublats::Vector{Sublat{T,E}}, model::Model; kw...) where {T,E} = System(sublats, Bravais{E}(), model; kw...)

function System(sublats::Vector{Sublat{T,E}}, bravais::Bravais{E,L} = Bravais{E}(), model::Model{Tv} = Model(); 
                dim::Val{E2} = Val(E), ptype::Type{T2} = T, htype::Type{Tv2} = Tv, kw...) where {Tv,T,E,L,Tv2,T2,E2}
    sdata = SublatsData(sublats)
    return convert(System{Tv2,T2,E2,L}, System(sublats, sdata, bravais, 
                   Operator{Tv2}(sublats, sdata, bravais, model)))
end

System(name::NameType; kw...) = systempresets[name](; kw...)
System(name::NameType, model::Model; kw...) = combine(System(name; kw...), model)

System(sys::System{Tv}, model::Model; kw...) where {Tv} = 
    System(sys.sublats, sys.bravais, convert(Model{Tv}, model); kw...)

Base.show(io::IO, sys::System{Tv,T,E,L}) where {Tv,T,E,L} = print(io, 
"System{$Tv,$T,$E,$L} : $(L)D system in $(E)D space
  Bravais vectors     : $(vectorsastuples(sys))
  Sublattice names    : $((sublatnames(sys)... ,))
  Sublattice orbitals : $((norbitals(sys)... ,))
  Total sites         : $(nsites(sys)) [$T]
  Total hoppings      : $(nlinks(sys)) [$Tv]
  Coordination        : $(coordination(sys))")  

#######################################################################
# System internal API
#######################################################################

vectorsastuples(sys::System) = vectorsastuples(sys.bravais.matrix)
vectorsastuples(br::Bravais) =  vectorsastuples(br.matrix)
vectorsastuples(mat::SMatrix{E,L}) where {E,L} = ntuple(l -> round.((mat[:,l]... ,), digits = 6), Val(L))

nsublats(sys::System) = length(sys.sublats)

sublatname(sys::System, s) = sys.sublatsdata.names[s]
sublatnames(sys::System) = sys.sublatsdata.names
sublatindex(sys::System, s) = sublatindex(sys.sublatsdata, s)

norbitals(sys::System) = sys.sublatsdata.norbitals
dim(s::System) = isempty(s.sublats) ? 0 : sum(dim, s.sublats)

nsites(sys::System) = sum(sys.sublatsdata.nsites)

nlinks(sys::System) = nlinks(sys.hamiltonian)
isunlinked(sys::System) = nlinks(sys) == 0
coordination(sys::System) = nlinks(sys.hamiltonian)/nsites(sys)

bravaismatrix(sys::System) = bravaismatrix(sys.bravais)
bravaismatrix(br::Bravais) = br.matrix

function uniquelinks(block::Block{Tv}, sys::System{Tv,T,E}) where {Tv,T,E}
    rdrs = [Tuple{SVector{E,T}, SVector{E,T}}[] for i in 1:nsublats(block), j in 1:nsublats(block)]
    bravais = bravaismatrix(sys)
    for (i, ((s1, s2), (target, source), (row, col), _)) in enumerate(BlockIterator(block))
        if row > col || !iszero(block.ndist)
            rdr = _rdr(sys.sublats[s2].sites[source], sys.sublats[s1].sites[target] + bravais * block.ndist)
            push!(rdrs[s1, s2], rdr)
        end
    end
    return rdrs
end

function boundingbox(sys::System{Tv,T,E}) where {Tv,T,E}
    bmin = zero(MVector{E, T})
    bmax = zero(MVector{E, T})
    foreach(sl -> foreach(s -> _boundingbox!(bmin, bmax, s), sl.sites), sys.sublats)
    return (SVector(bmin), SVector(bmax))
end

function _boundingbox!(bmin, bmax, site)
    bmin .= min.(bmin, site)
    bmax .= max.(bmax, site)
    return nothing
end

#######################################################################
# System external API
#######################################################################
"""
    modifysublats!(system; names, norbitals)

Change the names and/or number of orbitals of a system's sublattices

# Examples
```jldoctest
julia> modifysublats!(System(:honeycomb), names = (:P1, :P2), norbitals = (3, 1))
System{Complex{Float64},Float64,2,2} : 2D system in 2D space
  Bravais vectors     : ((0.5, 0.866025), (-0.5, 0.866025))
  Sublattice names    : (:P1, :P2)
  Sublattice orbitals : (3, 1)
  Total sites         : 2 [Float64]
  Total hoppings      : 0 [Complex{Float64}]
  Coordination        : 0.0
```
"""
modifysublats!

"""
    transform!(system::System, f::Function; sublats)

Change `system` in-place by moving positions `r` of sites in sublattices specified 
by `sublats` (all by default) to `f(r)`. Bravais vectors are also updated, but the 
system Hamiltonian is unchanged.

# Examples
```jldoctest
julia> transform!(System(:honeycomb, dim = Val(3)), r -> 2r + SVector(0,0,1))
System{Complex{Float64},Float64,3,2} : 2D system in 3D space
  Bravais vectors     : ((1.0, 1.732051, 0.0), (-1.0, 1.732051, 0.0))
  Sublattice names    : (:A, :B)
  Sublattice orbitals : (1, 1)
  Total sites         : 2 [Float64]
  Total hoppings      : 0 [Complex{Float64}]
  Coordination        : 0.0
```
"""
transform!

"""
    transform(system::System, f::Function; sublats)

Perform a `transform!` on a `deepcopy` of `system`, so that the result is completely
decoupled from the original. See `transform!` for more information.
```
"""
transform

"""
    combine(systems::System... [, model::Model])

Combine sublattices of a set of systems (giving them new names if needed) and optionally 
apply a new `model` that describes their coupling.

# Examples
```jldoctest
julia> combine(System(:honeycomb), System(:honeycomb), Model(hopping(1, sublats = (:A, :B4))))
System{Complex{Float64},Float64,2,2} : 2D system in 2D space
  Bravais vectors     : ((0.5, 0.866025), (-0.5, 0.866025))
  Sublattice names    : (:A, :B, :A3, :B4)
  Sublattice orbitals : (1, 1, 1, 1)
  Total sites         : 4 [Float64]
  Total hoppings      : 6 [Complex{Float64}]
  Coordination        : 1.5
```
"""
combine

"""
    grow(system::System{Tv,T,E,L}; supercell = SMatrix{L,0,Int}(), region = r -> true)

Transform `system` into another system with a different supercell, so that the new
Bravais matrix is `br2 = br * supercell`, and only sites with `region(r) == true` in the
unit cell are included. 

`supercell` can be given as an integer matrix, a single integer `s` (`supercell = s * I`),
a single `NTuple{L,Int}` (`supercell` diagonal), or a tuple of  `NTuple{L,Int}`s (`supercell` 
columns). Note that if the new system dimension `L2` is smaller than the original, a bounded 
`region` function should be provided to determine the extension of the remaining dimensions.

# Examples
```jldoctest
julia> grow(System(:triangular, Model(hopping(1))), supercell = (3, -3), region = r-> 0 < r[2] < 12)
System{Complex{Float64},Float64,2,2} : 2D system in 2D space
  Bravais vectors     : ((1.5, 2.598076), (1.5, -2.598076))
  Sublattice names    : (:S1,)
  Sublattice orbitals : (1,)
  Total sites         : 3 [Float64]
  Total hoppings      : 6 [Complex{Float64}]
  Coordination        : 2.0
```
"""
grow

"""
    hamiltonian(system; k, kphi)

Returns the Bloch Hamiltonian of an `L`-dimensional `system` in `E`-dimensional space at 
a given `E`-dimensional Bloch momentum `k`, or alternatively `L`-dimensional normalised 
Bloch phases `kphi = k*B/2π`, where `B` is the system's Bravais matrix.
By default the Hamiltonian at zero momentum (Gamma point) is returned. For `0`-dimensional 
systems, the Bloch Hamiltonian is simply the Hamiltonian of the system.

# Examples
```jldoctest
julia> hamiltonian(System(:honeycomb, Model(hopping(1))), momentum = (0,1))
2×2 SparseMatrixCSC{Complex{Float64},Int64} with 4 stored entries:
  [1, 1]  =  2.59144+0.0im
  [2, 1]  =  2.29572-1.52352im
  [1, 2]  =  2.29572+1.52352im
  [2, 2]  =  2.59144+0.0im
```
"""
hamiltonian

#######################################################################
# LazySystem
#######################################################################
# """
#     LazySystem(system, directives...)

# Build a `LazySystem{T,E,L,EL}` by applying `directives` on `system`, but
# only at evaluation time (when computing the Hamiltonian or a KPM iteration). 

# Typical use involves `OnsiteModifier`, `HoppingModifier` and `Region` directives
# to implement magnetic flux, disorder or finite regions in matrix-free methods or 
# when needing to sweep parameter space without recomputing the base `system`.
# """


# function System(l::Lattice, m::Model)
# 	hop = hamiltonianoperator(l, m)
# 	vop = velocityoperator(hop)
# 	return System(l, m, hop, vop)
# end

# System(name::Symbol) = System(Preset(name))
# System(preset::Preset) = haskey(systempresets, preset.name) ? systempresets[preset.name](; preset.kwargs...) : 
#     System(Lattice(preset), Model(Onsite(0), Hopping(1)))

# #######################################################################
# # Display
# #######################################################################

# function Base.show(io::IO, sys::System{T,E,L}) where {T,E,L}
#     print(io, "System{$T,$E,$L} : $(L)D system in $(E)D space with $T sites.
#     Bravais vectors : $(vectorsastuples(sys.lattice))
#     Number of sites : $(nsites(sys.lattice))
#     Sublattice names : $((sublatnames(sys.lattice)... ,))
#     Unique Links : $(nlinks(sys.lattice))
#     Model with sublattice site dimensions $((sys.model.dims...,)) (default $(sys.model.defdim))
#     $(sys.hbloch)")
# end

# #######################################################################
# # build hamiltonian!
# #######################################################################

# velocityoperator(hbloch::Operator{T,L}; kn = zero(SVector{L,Int}), axis = 1) where {T,L} =
# 	gradient(hbloch; kn = kn, axis = axis)
# function hamiltonianoperator(lat::Lattice{T,E,L}, model::Model) where {T,E,L}
#     dimh = hamiltoniandim(lat, model)

#     I = Int[]
#     J = Int[]
#     V = Complex{T}[]
#     Vn = [Complex{T}[] for k in 1:length(lat.links.interlinks)]
#     Voffsets = Int[]

#     hbloch!(I, J, V, model, lat, lat.links.intralink, false)
#     push!(Voffsets, length(V) + 1)

#     for (k, interlink) in enumerate(lat.links.interlinks)
#         hbloch!(I, J, Vn[k], model, lat, interlink, true)
#         append!(V, Vn[k])
#         push!(Voffsets, length(V) + 1)
#     end

#     ndist = [interlink.ndist for interlink in lat.links.interlinks]
#     workspace = SparseWorkspace{T}(dimh, length(I))

#     mat = sparse!(I, J, V, dimh, workspace)

#     return Operator(I, J, V, Voffsets, Vn, ndist, workspace, mat)
# end

# function hamiltoniandim(lat, model)
#     dim = 0
#     for (s, d) in enumerate(sublatdims(lat, model))
#         dim += nsites(lat.sublats[s]) * d
#     end
#     return dim
# end
# hamiltoniandim(sys::System) = size(sys.hbloch.matrix, 1)
# hamiltoniandim(h::AbstractMatrix) = size(h, 1)

# sparse!(I, J, V, dimh, workspace::SparseWorkspace) =
#     sparse!(I, J, V, dimh, dimh, +,
#             workspace.klasttouch, workspace.csrrowptr,
#             workspace.csrcolval, workspace.csrnzval)

# sparse!(h, I, J, V, dimh, workspace::SparseWorkspace) =
#     sparse!(I, J, V, dimh, dimh, +,
#             workspace.klasttouch, workspace.csrrowptr,
#             workspace.csrcolval, workspace.csrnzval,
#             h.colptr, h.rowval, h.nzval)

# updateoperatormatrix!(op) = sparse!(op.matrix, op.I, op.J, op.V, size(op.matrix, 1), op.workspace)

# function hbloch!(I, J, V, model, lat, ilink, isinter)
#     sdims = sublatdims(lat, model) # orbitals per site of each sublattice
#     coloffsetblock = 0
#     for (s1, subcols) in enumerate(sdims)
#         rowoffsetblock = 0
#         for (s2, subrows) in enumerate(sdims)
#             if !isinter && s1 == s2
#                 appendonsites!(I, J, V, rowoffsetblock, lat.sublats[s1], onsite(model, s1), Val(subrows))
#             end
#             if isvalidlink(isinter, (s1, s2))
#                 appendhoppings!(I, J, V, (rowoffsetblock, coloffsetblock), ilink.slinks[s2, s1], hopping(model, (s2, s1)), Val(subrows), Val(subcols), !isinter)
#             end
#             rowoffsetblock += subrows * nsites(lat.sublats[s2])
#         end
#         coloffsetblock += subcols * nsites(lat.sublats[s1])
#     end
#     return nothing
# end

# appendonsites!(I, J, V, offsetblock, sublat, ons::NoOnsite, ::Val) = nothing
# function appendonsites!(I, J, V, offsetblock, sublat, ons, ::Val{subrows}) where {subrows}
#     offset = offsetblock
#     for r in sublat.sites
#         o = ons(r, Val(subrows))
#         for inds in CartesianIndices(o)
#             append!(I, offset + inds[1])
#             append!(J, offset + inds[2])
#         end
#         append!(V, real(o))
#         offset += subrows
#     end
#     return nothing
# end

# appendhoppings!(I, J, V, (rowoffsetblock, coloffsetblock), slink, hop::NoHopping, ::Val, ::Val, symmetrize) = nothing
# function appendhoppings!(I, J, V, (rowoffsetblock, coloffsetblock), slink, hop, ::Val{subrows}, ::Val{subcols}, symmetrize) where {subrows, subcols}
#     posstart = length(I)
#     for src in sources(slink), (target, rdr) in neighbors_rdr(slink, src)
#         rowoffset = (target - 1) * subrows
#         coloffset = (src - 1) * subcols
#         h = hop(rdr, Val(subrows), Val(subcols))
#         for inds in CartesianIndices(h)
#             append!(I, rowoffsetblock + rowoffset + inds[1])
#             append!(J, coloffsetblock + coloffset + inds[2])
#         end
#         append!(V, h)
#     end
#     posend = length(I)
#     if symmetrize
#         # We assume only uniquelinks in intralinks. We add the hermitian conjugate part
#         # This should be removed if isvalidlink does not filter out half of the links
#         append!(I, view(J, (posstart+1):posend))
#         append!(J, view(I, (posstart+1):posend))
#         sizehint!(V, posend + (posend - posstart))
#         for k in (posstart+1):posend
#             @inbounds push!(V, conj(V[k]))
#         end
#     end
#     return nothing
# end

# function blochphases(k, sys::System{T,E}) where {T,E}
# 	length(k) == E || throw(DimensionMismatch("The dimension of the Bloch vector `k` should math the embedding dimension $E"))
# 	return transpose(bravaismatrix(sys.lattice)) * SVector(k) / (2pi)
# end

# function hamiltonian!(sys::System{T,E,L}; k = zero(SVector{E,T}), kn = blochphases(k, sys), intracell::Bool = false) where {T,E,L}
# 	length(kn) == L || throw(DimensionMismatch("The dimension of the normalized Bloch phases `kn` should match the lattice dimension $L"))
# 	insertblochphases!(sys.hbloch, SVector{L,T}(kn), intracell)
#     updateoperatormatrix!(sys.hbloch)
#     return sys.hbloch.matrix
# end

# hamiltonian(sys; kw...) = copy(hamiltonian!(sys; kw...))

# function velocity!(sys::System{T,E,L}; k = zero(SVector{E,T}), kn = blochphases(k, sys), axis::Int = 1) where {T,E,L}
# 	0 <= axis <= max(L, 1) || throw(DimensionMismatch("Keyword `axis` should be between 0 and $L, the lattice dimension"))
# 	length(kn) == L || throw(DimensionMismatch("The dimension of the normalized Bloch phases `kn` should match the lattice dimension $L"))
# 	insertblochphases!(sys.vbloch, SVector{L,T}(kn), axis)
# 	updateoperatormatrix!(sys.vbloch)
# 	return sys.vbloch.matrix
# end

# velocity(sys; kw...) = copy(velocity!(sys; kw...))
