#######################################################################
# System
#######################################################################
"""
    System(sublats::Sublat... [, br::Bravais[, model::Model]]; dim::Val, postype, hamtype)

Build a `System{E,L,T,Tv}` of `L` dimensions in `E`-dimensional embedding
space and composed of `T`-typed sites and `Tv`-typed Hamiltonian. See
`Sublat`, `Bravais` and `Model` for syntax.  To override the embedding dimension `E`, use 
keyword `dim = Val(E)`. Similarly, override types `T` and `Tv` with `ptype = T` and 
`htype = Tv`.

    System(presetname::$(NameType)[, model]; kw...)

Build a system from a given preset, and optionally add a `model` to it. Preset 
names are defined in the dictionary `Elsa.systempresets`, together with their 
allowed keyword arguments `kw`. These typically include `norbitals` for sublattices.

    System(sys::System, model::Model)

Build a new system with the same sublattices and bravias vectors as `sys`
but replacing its Hamiltonian with a new `model`.

# Examples
```jldoctest
julia> System(Sublat((1,0), (0,1); name = :C, norbitals = 2), 
              Sublat((0.5,0.5); name = :D), 
              Bravais((1,0), (0,2)),
              Model(Hopping((r,dr) -> @SMatrix[0.1; dr[1]], sublats = (:C, :D), range = 1)))
System{2,2,Float64,Complex{Float64}} : 2D system in 2D space
  Bravais vectors     : ((1.0, 0.0), (0.0, 2.0))
  Sublattice names    : (:C, :D)
  Sublattice orbitals : (2, 1)
  Total sites         : 3 [Float64]
  Total hoppings      : 8 [Complex{Float64}]
  Coordination        : 2.6666666666666665

julia> System(:honeycomb, Model(Hopping(@SMatrix[1 2; 0 1], sublats = (1,2))), 
              dim = Val(3), htype = Float32, norbitals = 2)
System{3,2,Float64,Float32} : 2D system in 3D space
  Bravais vectors     : ((0.5, 0.866025, 0.0), (-0.5, 0.866025, 0.0))
  Sublattice names    : (:A, :B)
  Sublattice orbitals : (2, 2)
  Total sites         : 2 [Float64]
  Total hoppings      : 6 [Float32]
  Coordination        : 3.0

julia> Tuple(keys(Elsa.systempresets))
(:bcc, :cubic, :honeycomb, :linear, :graphene_bilayer, :square, :triangular)
```

# See also:

    `Model`, `Onsite`, `Hopping`, `grow`, `transform`, `combine`
"""
struct System{E,L,T,Tv,S<:SystemInfo,EL}
    lattice::Lattice{E,L,T,EL}
    hamiltonian::Operator{Tv,L}
    velocity::Operator{Tv,L}
    sysinfo::S
end

System(s1::Sublat, s2...; kw...) = System(_collectlattice((s1,), s2...)...; kw...)
_collectlattice(ss::Tuple, s1::Sublat, s2...) = _collectlattice((ss..., s1), s2...)
_collectlattice(ss::Tuple, b::Bravais, m::Model = Model()) = ((b, ss...), m)
_collectlattice(ss::Tuple, m::Model = Model()) = (ss, m)
System(latparts::Tuple, m::Model; kw...) = System(Lattice(latparts...; kw...), m; kw...)

function System(lat::Lattice{E,L,T}, model::Model{Tv} = Model(); 
                htype::Type{Tv2} = Tv, kw...) where {E,L,T,Tv,Tv2}
    actualmodel = convert(Model{Tv2}, model)
    sysinfo = SystemInfo(lat, actualmodel)
    hamiltonian = Operator(lat, sysinfo)
    velocity = boundaryoperator(hamiltonian)
    return System(lat, hamiltonian, velocity, sysinfo)
end

System(name::NameType; kw...) = systempresets[name](; kw...)
System(name::NameType, model::Model; kw...) = combine(System(name; kw...), model)

System(sys::System{E,L,T,Tv}, model::Model; kw...) where {E,L,T,Tv} = System(sys.lattice, convert(Model{Tv}, model); kw...)

Base.show(io::IO, sys::System{E,L,T,Tv}) where {E,L,T,Tv} = print(io, 
"System{$E,$L,$T,$Tv} : $(L)D system in $(E)D space
  Bravais vectors     : $(vectorsastuples(sys))
  Sublattice names    : $((sublatnames(sys)... ,))
  Sublattice orbitals : $((norbitals(sys)... ,))
  Total sites         : $(nsites(sys)) [$T]
  Total hoppings      : $(nlinks(sys)) [$Tv]
  Coordination        : $(coordination(sys))")

# Treat System as scalar
Broadcast.broadcastable(sys::System) = Ref(sys)

#######################################################################
# System internal API
#######################################################################

vectorsastuples(sys::System) = vectorsastuples(sys.lattice.bravais.matrix)
vectorsastuples(mat::SMatrix{E,L}) where {E,L} = ntuple(l -> round.((mat[:,l]... ,), digits = 6), Val(L))

nsublats(sys::System) = length(sys.lattice.sublats)

sublatindex(sys::System, s) = sublatindex(sys.sysinfo.namesdict, s)
sublatindex(s::SystemInfo, name) = sublatindex(s.namesdict, name)
sublatindex(d::Dict, name::NameType) = d[name]
sublatindex(s::Dict, i::Integer) = Int(i)

sublatname(sys::System, s) = sys.sysinfo.names[s]
sublatnames(sys::System) = _parsename.(sys.sysinfo.names)
_parsename(name::Symbol) = (sname = String(name); first(sname) == '_' ? Base.parse(Int, sname[2:end]) : name)

norbitals(sys::System) = sys.sysinfo.norbitals
dim(s::System) = isempty(s.sublats) ? 0 : sum(dim, s.sublats)

nsites(sys::System) = sum(sys.sysinfo.nsites)

nlinks(sys::System) = nlinks(sys.hamiltonian)
isunlinked(sys::System) = nlinks(sys) == 0
coordination(sys::System) = nlinks(sys.hamiltonian)/nsites(sys)
site(sys::System, s::Int, i::Int) = sys.lattice.sublats[s].sites[i]

function boundingbox(sys::System{E,L,T}) where {E,L,T}
    bmin = zero(MVector{E,T})
    bmax = zero(MVector{E,T})
    foreach(sl -> foreach(s -> _boundingbox!(bmin, bmax, s), sl.sites), sys.lattice.sublats)
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
    transform!(system::System, f::Function; sublats)

Change `system` in-place by moving positions `r` of sites in sublattices specified 
by `sublats` (all by default) to `f(r)`. Bravais vectors are also updated, but the 
system Hamiltonian is unchanged.

    system |> transform(f::Function; kw...)

Functional syntax, equivalent to `transform(system, f; kw...)`

# Examples
```jldoctest
julia> transform!(System(:honeycomb, dim = Val(3)), r -> 2r + SVector(0,0,1))
System{3,2,Float64,Complex{Float64}} : 2D system in 3D space
  Bravais vectors     : ((1.0, 1.732051, 0.0), (-1.0, 1.732051, 0.0))
  Sublattice names    : (:A, :B)
  Sublattice orbitals : (0, 0)
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

    system |> combine(model)

Functional syntax, equivalent to `combine(system, model)`, which adds `model` to an existing
system.

# Examples
```jldoctest
julia> combine(System(:honeycomb), System(:honeycomb), Model(Hopping(1, sublats = (:A, 4))))
System{2,2,Float64,Complex{Float64}} : 2D system in 2D space
  Bravais vectors     : ((0.5, 0.866025), (-0.5, 0.866025))
  Sublattice names    : (:A, :B, 3, 4)
  Sublattice orbitals : (1, 0, 0, 1)
  Total sites         : 4 [Float64]
  Total hoppings      : 6 [Complex{Float64}]
  Coordination        : 1.5
```
"""
combine

"""
    grow(system::System{E,L}; supercell = SMatrix{L,0,Int}(), region = r -> true)

Transform an `L`-dimensional `system` into another `L2`-dimensional system with a different 
supercell, so that the new Bravais matrix is `br2 = br * supercell`, and only sites with 
`region(r) == true` in the unit cell are included. 

`supercell` can be given as an integer matrix `s::SMatrix{L,L2,Int}`, a single integer 
`s::Int` (`supercell = s * I`), a single `NTuple{L,Int}` (`supercell` diagonal), or a tuple 
of  `NTuple{L,Int}`s (`supercell` columns). Note that if the new system dimension `L2` is 
smaller than the original, a bounded `region` function should be provided to determine the 
extension of the remaining dimensions.

    system |> grow(region = f, supercell = s)

Functional syntax, equivalent to `grow(system; region = f, supercell = s)

# Examples
```jldoctest
julia> grow(System(:triangular, Model(Hopping(1))), supercell = (3, -3), region = r-> 0 < r[2] < 12)
System{2,2,Float64,Complex{Float64}} : 2D system in 2D space
  Bravais vectors     : ((1.5, 2.598076), (1.5, -2.598076))
  Sublattice names    : (1,)
  Sublattice orbitals : (1,)
  Total sites         : 3 [Float64]
  Total hoppings      : 6 [Complex{Float64}]
  Coordination        : 2.0
```

# See also

    'Region`
"""
grow

"""
    bound(system::System{E,L,T}; except = ()))

Remove the periodicity of the `system`'s lattice along the specified `axes`, dropping
Bloch Hamiltonian harmonics accordingly.

    system |> bound(; kw...)

Functional syntax, equivalent to `bound(system; kw...)``

# Examples
```jldoctest
julia> bound(System(:cubic), except = (1, 3))
System{3,2,Float64,Complex{Float64}} : 2D system in 3D space
  Bravais vectors     : ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
  Sublattice names    : (1,)
  Sublattice orbitals : (0,)
  Total sites         : 1 [Float64]
  Total hoppings      : 0 [Complex{Float64}]
  Coordination        : 0.0

julia> System(:triangular) |> bound()
System{2,0,Float64,Complex{Float64}} : 0D system in 2D space
  Bravais vectors     : ()
  Sublattice names    : (1,)
  Sublattice orbitals : (0,)
  Total sites         : 1 [Float64]
  Total hoppings      : 0 [Complex{Float64}]
  Coordination        : 0.0
```
"""
bound

"""
    hamiltonian(system; k, ϕn)

Return the Bloch Hamiltonian of an `L`-dimensional `system` in `E`-dimensional space at 
a given `E`-dimensional Bloch momentum `k`, or alternatively `L`-dimensional normalised 
Bloch phases `ϕn = k*B/2π`, where `B` is the system's Bravais matrix.
By default the Hamiltonian at zero momentum (Gamma point) is returned. For `0`-dimensional 
systems, the Bloch Hamiltonian is simply the Hamiltonian of the system.

Important note: for performance reasons, `hamiltonian` reuses always the same preallocated 
sparse matrix `system.hamiltonian.matrix`. Hence doing `h1 = hamiltonian(sys, k = k1)` and 
then `h2 = hamiltonian(sys, k = k2)` overwrites `h1`, so that `h1 === h2`.

# Examples
```jldoctest
julia> hamiltonian(System(:honeycomb, Model(Hopping(1))), ϕn = (0,0.5))
2×2 SparseArrays.SparseMatrixCSC{Complex{Float64},Int64} with 4 stored entries:
  [1, 1]  =  -2.0+0.0im
  [2, 1]  =  1.0-1.22465e-16im
  [1, 2]  =  1.0+1.22465e-16im
  [2, 2]  =  -2.0+0.0im
```

# See also

    'velocity`
"""
hamiltonian

"""
    velocity(system::System{E,L}, axis = missing; k, ϕn)

Return the velocity operator `∂H(k)` along the specified crystallographic `axis`, which can
be an integer from 1 to `L` (unit axis), an `SVector{L}` or an `NTuple{L}`. If no `axis` is 
given, a tuple of all velocity operators along each of the `L` axis is returned. See 
`hamiltonian` for details on the `k` and `ϕn` keywords.

Important note: for performance reasons, `velocity` reuses the same preallocated sparse 
matrix `system.velocity.matrix`. Hence doing `h1 = velocity(sys, axis1, k = k1)` and 
then `h2 = velocity(sys, axis2, k = k2)` overwrites `h1`, so that `h1 === h2`. An exception
is the case `axis = missing`, wherin each returned velocity operator is an independent copy.

# Examples
```jldoctest
julia> velocity(System(:honeycomb, Model(Hopping(1))), 1; ϕn = (0,0.5))
2×2 SparseArrays.SparseMatrixCSC{Complex{Float64},Int64} with 4 stored entries:
  [1, 1]  =  -2.0+0.0im
  [2, 1]  =  1.0-1.22465e-16im
  [1, 2]  =  1.0+1.22465e-16im
  [2, 2]  =  -2.0+0.0im
```

# See also

    'hamiltonian`
"""
velocity

"""
    bravaismatrix(system)

Return the Bravais matrix of a system. Its columns are the Bravais vectors.
"""
bravaismatrix(sys::System) = sys.lattice.bravais.matrix
bravaismatrix(lat::Lattice) = lat.bravais.matrix
bravaismatrix(br::Bravais) = br.matrix

"""
    sitepositions(system[, sublat::Union{Integer, $NameType}[, siteindex::Int]])
    sitepositions(system, (sublat, siteindex))

Return a vector with all the site positions of `system`, or only those in `sublat` 
or with a given `siteindex`, if specified.
"""
sitepositions(sys::System, s, i) = sys.lattice.sublats[sublatindex(sys, s)].sites[i]
sitepositions(sys::System, (s, i)::Tuple) = sitepositions(sys, s, i)
sitepositions(sys::System, s::Union{NameType,Integer}) = sys.lattice.sublats[sublatindex(sys, s)].sites
sitepositions(sys::System) = [sl.sites for sl in sys.lattice.sublats]

"""
    neighbors(system, sublat, siteindex; targetsublat = missing)
    neighbors(system, (sublat, siteindex); targetsublat = missing)

Return a vector with `(targetsublat, targetsiteindex)` of all the neighbors of input site 
(with `sublatindex` in sublattice `sublat`). A non-missing `targetsublat` restricts the search
to a specific sublat.

# Examples
```jldoctest
julia> sys = System(:honeycomb, Model(Hopping(1))) |> grow(region = Region(:square, 2))
System{2,0,Float64,Complex{Float64}} : 0D system in 2D space
  Bravais vectors     : ()
  Sublattice names    : (:A, :B)
  Sublattice orbitals : (1, 1)
  Total sites         : 10 [Float64]
  Total hoppings      : 50 [Complex{Float64}]
  Coordination        : 5.0

julia> neighbors(sys, 1, 4)
4-element Array{Tuple{Int64,Int64},1}:
 (1, 2)
 (1, 5)
 (2, 3)
 (2, 5)

julia> neighbors(sys, 1, 4, targetsublat = :A)
2-element Array{Tuple{Int64,Int64},1}:
 (1, 2)
 (1, 5)

```
"""
neighbors(sys, (sublat, siteindex)::Tuple; kw...) = neighbors(sys, sublat, siteindex; kw...)
function neighbors(sys, sublat, siteindex; targetsublat = missing)
    h = sys.hamiltonian.matrix
    rows = rowvals(h)
    source = torow(siteindex, sublat, sys.sysinfo)
    targets = Tuple{Int,Int}[]
    tsublatindex = targetsublat === missing ? 0 : sublatindex(sys, targetsublat)
    for ptr in nzrange(h, source)
        target, offset, tsublat = tosite(rows[ptr], sys.sysinfo)
        offset == 0 && (targetsublat === missing || tsublatindex === tsublat) && 
            (tsublat != sublat || target != source) && push!(targets, (tsublat, target))
    end
    return sort!(targets)
end


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
