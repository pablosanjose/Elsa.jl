# Getting started

`Elsa.jl` is a Julia package to help define tight-binding systems, construct their Hamiltonians, and obtain various quantities of physical interest about them, e.g. transport properties and electronic structure.

The basic workflow is as follows. One can define a set of site positions in space, together with a number of orbitals per site and a name. These constitute a sublattice (of type `Sublat`), which share the same name and number of orbitals. A collection of `Sublat`s, together with a set of Bravais vectors if the system is an infinite, periodic lattice, constitute a `Lattice`. A `System` is a `Lattice` together wih a `Model`, that defines how sites in different `Sublat`s are coupled (the hoppings and onsite energies in a tight-binding language). With the `Model` the `System` establishes links between sites and stores a Hamiltonian (of type `Operator`), which can be used to compute all sort of quantities using various algorithms in `Elsa.jl`.

The following is a brief description of basic functionality to get started.

## Installation
You need Julia 1.0 or higher to use `Elsa.jl`. Until I register the package it should be installed as follows
```julia
julia>] add https://github.com/pablosanjose/Elsa.jl 
```
Then, to start using it in a Julia session do `using Elsa`

Optionally, you can also install the associated `ElsaPlots.jl` package to visualise systems, bandstructures, etc.
```julia
julia>] add https://github.com/pablosanjose/ElsaPlots.jl 
```


## Basic concepts

These are some of the basic types in `Elsa.jl`:

- `Sublat{T,E}`: a set of sites, given as a `Vector` of positions `SVector{E,T}` (`E` is the embedding dimension, and `T<:Number` is the type of coordinates)
    - We don't *need* to enter positions as `SVector`s when using `Elsa.jl`. Tuples will do. 
    - *Example*: `Sublat((1.0,0), (0,1))` will promote the two site positions to `SVector{Float64,2}`
    - We can give a name to a `Sublat` with `Sublat((1,0), (0,1)), name = :A)`. Note that names are `Symbol`s, which in Julia are preceded with a `:`

- `Bravais`: a set of Bravais vectors of a lattice
    - We can provide a set of vectors or a matrix with Bravais vectors as columns
    - *Example*: `Bravais((1,0), (0,1))` are the Bravais vectors of a square lattice in 2D.
    - The number of Bravais vectors defines the "dimensions" `L` of a lattice (2 above), not to be confused with the dimensions `E` of the embedding space.

- `Lattice`: a set of `Sublats` bundled together with a `Bravais` matrix. One typically doesn't need to build a `Lattice` explicitly, and can instead build a `System` directly by using its constructor, which can take `Sublats`, `Bravais` and a `Model`.

- `Model`: A tight-binding model given as a set of `Hopping` and `Onsite` instructions `Model(Onsite(...), Hopping(...), ...)` used to produce the (Bloch) Hamiltonian of a given system. Any hopping or onsite energy not address by the spedified `Hopping` or `Onsite` instructions in a `Model` will be taken as zero in the resulting Hamiltonian. Hence and empty `Model()` will produce just a Hamiltonian full of zeros when passed to a `System`. The syntax of `Hopping` and `Onsite` terms are as follows:
    - `Hopping(h, sublats = ((s1, s2), (s3, s4), ...), ndists = (n1, n2, ...)), range = 1)`
    - `Onsite(o, sublats = (s1, s2, ...))`
    - Here `h` and `o` are either constants, matrices in orbital space (use e.g. `@SMatrix[1 2; 3 4]` for performance) or functions of position. Onsite functions take one position (e.g. `o = r -> @SMatrix[r[1] 0; 0 2r[2]]`), and hopping functions take two positions, the center of the link between `r1` and `r2`, `r = (r1 + r2)/2`, and the link vector `dr = r2 - r1` (e.g. `h = (r,dr) -> exp(i dot(A(r), dr))` for a Peierls phase)
    - `sublats` `s1, s2,...` are either numbers or `Sublat` names to which the onsite or hopping is to be applied.
    - `ndists` `n1, n2...` in `Hopping` are integer vectors or tuples that define the Bravais distance between two unit cells to be linked. In other words, to link any given unit cell with another at distance `bravais * n1` specify `n1` in `ndists`.
    - `range` is the maximum Euclidean distance (`1` by default) at which to look for neighbor candidates of each site.
    - Hermitian conjugate pairs are added for any hopping that is specified, so that the resulting Hamiltonian is Hermitian.

- `System`: the basic object whose properties we want to compute. Its a combination of a `Lattice` (itself a combination of one or several `Sublat`s and a `Bravais` matrix) and a tight-binding `Model` that becomes translated into a Hamiltonian of type `Operator`.
    - The canonical constructor is `System(sublats..., bravais, model)`. This collects `sublats` and `bravais` into a `Lattice`, applies `model` to it, and bundles the resulting hamiltonian with the lattice.
    - Alternatively we can use presets, defined in the `Dict` `systempresets`. *Example*: `System(:honeycomb, Model(Hopping(1)))`
    - We can override the type `T` of site positions, the element type `Tv` of the Hamiltonian and the embedding dimension `E` using keyword arguments in the `System` constructor, as in `System(:honeycomb, dim = Val(E), ptype = T, htype = Tv)`.

## System API

A number of functions are available to easily build complex systems incrementally (see examples below). These are 
- `grow(system; supercell = ..., region = ...)` to expand the unit cell of a `system` with Bravais vectors `bravais` so that the result has bravais vectors `bravais * supercell` (could be less vectors than the original). `region` is a boolean function of spatial position that specifies if a site at `r` in the unit cell should be included in the result.
- `transform!(system, r -> f(r); sublats = ...)` change `system` in-place by moving sites at `r` to `f(r)`. If `sublats` (names or numbers) are specified, the transformation is restricted to those sublattices. Bravais vectors are updated too, but no change to the Hamiltonian is performed. See also `transform` for not-in-place transformations.
- `combine(systems...[, model])` to combine the sublattices of all `systems` into a single system, giving them new unique names if necessary. The tight-binding structure of each system is preserved. Couplings between subsystems can be added using `model`. Note that `combine` can also be used to add a given `model` to a single existing system.
- `hamiltonian(system; k, kphi)` gives the Hamiltonian of a closed system (`L == 0`) or the Bloch Hamiltonian of an unbounded system, at wavevector `k`, or alternatively at normalised Bloch phases `kphi = k*B/2π` (`B` is the Bravais matrix). This gives a plain `SparseMatrixCSC`, computed from a more complex object `system.hamiltonian`, of type `Operator`.

One can use a functional for of all the above to chain a number of operations. The two following instructions would then be equivalent
```julia
System(:honeycomb) |> grow(region = Region(:circle, 3))
```
```julia
grow(System(:honeycomb), region = Region(:circle, 3))
```

Lastly, one can replace the Hamiltonian of a system `sys` using a different model like this: `System(sys, model)`. This preserves all sublattices and Bravais vectors, but recomputes the Hamiltonian (unlike `combine` which adds to the existing Hamiltonian).

## Visualising a system
`ElsaPlots.jl` can be used to visualise a number of `Elsa.jl` objects in 2D or 3D using `Makie.jl`. For example, we can visualise the lattice and links of `sys` with

```julia
julia> using ElsaPlots

julia> plot(sys)
```

Note there is a significant load delay and time-to-first-plot due to the precompilation of the underlying `Makie.jl` machinery used by `ElsaPlots.jl` (unless you have precompiled `Makie.jl` into your system image).

Other objects produced by `Elsa.jl` can likewise be visualised using `plot`.

# Examples
A series of simple usage examples follow of increasing complexity.

## Example 1: nearest-neighbor graphene lattice
To get started straight away let us show how to build a simple, infinite graphene lattice, with nearest neighbor hoppings. For the sake of clarity I describe below the fully manual way, which is more verbose and wouldn't be the way to do things in practice.

- Define the lattice constant in nanometers
```julia
a0 = 0.246;
```
- Create a sublattice of type `Sublat <: BuildInstruction` that contains a single site at position `(0.0, -0.5/sqrt(3.0) * a0)`, and name it sublattice `:A`
```julia
subA = Sublat((0.0, -0.5/sqrt(3.0) * a0); name = :A);
```
- Create another sublattice `:B`, again with a single site, but at `(0.0, 0.5/sqrt(3.0)`
```julia
subB = Sublat((0.0, 0.5/sqrt(3.0) * a0); name = :B);
```
- Define Bravais vectors `(cos(pi/3), sin(pi/3) .* a0` and `(-cos(pi/3), sin(pi/3)) .* a0`
```julia
br = Bravais(( cos(pi/3), sin(pi/3)) .* a0, 
             (-cos(pi/3), sin(pi/3)) .* a0);
```
- Create a `Model` with a hopping of value `1` between sites at up to a distance `a0/√3`
```julia
model = Model(Hopping(1, range = a0/√3));
```
- Pass all these directives to the `System` constructor.
```julia
julia> sys = System(subA, subB, br, model)
System{2,2,Complex{Float64},Float64} : 2D system in 2D space
  Bravais vectors     : ((0.123, 0.213042), (-0.123, 0.213042))
  Sublattice names    : (:A, :B)
  Sublattice orbitals : (1, 1)
  Total sites         : 2 [Float64]
  Total hoppings      : 6 [Complex{Float64}]
  Coordination        : 3.0
```
The resulting object is of type `System{E,L,T,Tv}`, where `E = 2` is the embedding dimension (2D space in this example), `L = 2` is the lattice dimension (two because graphene is a 2D Bravais lattice, with two bravais vectors), `T = Float64` is the numeric type of positions and `Tv = Complex{Float64}` is the type of the Hamiltonian elements. The reported coordination is the average number of links out of each site (3 in graphene with nearest neighbors). The number `Total hoppings` indicates how many links exist within the unit cell and between the unit cell and other linked unit cells. In this case this is six, see the following sketch.
```
 \/
 ||
 /\
```

### Example 1 using presets

The above can be streamlined by using built-in presets. System presets are just pre-defined `System`s that are included in `Elsa.jl` for convenience. These range from simple Bravais lattice with an empty `Model()` to more complicated twisted bilayer graphene system that can be customised through a set of keyword parameters.

(Note that there are other non-system presets, such as `Region` presets that provide pre-defined regions in space, to be introduced in Example 2. Here we just focus of *system* presets.)

Currently available system presets are (the list will keep growing):
```julia
julia> Tuple(keys(Elsa.systempresets))
(:bcc, :cubic, :honeycomb, :linear, :graphene_bilayer, :square, :triangular)
```

We can use a system preset by passing its name to the `System` builder. 
```julia
julia> System(:honeycomb)
System{2,2,Float64,Complex{Float64}} : 2D system in 2D space
  Bravais vectors     : ((0.5, 0.866025), (-0.5, 0.866025))
  Sublattice names    : (:A, :B)
  Sublattice orbitals : (0, 0)
  Total sites         : 2 [Float64]
  Total hoppings      : 0 [Complex{Float64}]
  Coordination        : 0.0
```

We can replace the model of a system preset, and/or change sublattice orbital count as follows
```julia
julia> System(:triangular, Model(Hopping(@SMatrix[0 2; 0 0])))
System{2,2,Float64,Complex{Float64}} : 2D system in 2D space
  Bravais vectors     : ((0.5, 0.866025), (-0.5, 0.866025))
  Sublattice names    : (1,)
  Sublattice orbitals : (2,)
  Total sites         : 1 [Float64]
  Total hoppings      : 6 [Complex{Float64}]
  Coordination        : 6.0
```

Some more complex system presets have additional keyword arguments that can be specified just as above. As an example, a periodic, twisted graphene bilayer with a moiré pattern index `twistindex = 31` can be built with the following preset
```julia
julia> System(:graphene_bilayer, twistindex = 31)
System{3,2,Float64,Complex{Float64}} : 2D system in 3D space
  Bravais vectors     : ((-0.0, 13.422225, 0.0), (-11.623988, 6.711113, 0.0))
  Sublattice names    : (:Ab, :Bb, :At, :Bt)
  Sublattice orbitals : (1, 1, 1, 1)
  Total sites         : 11908 [Float64]
  Total hoppings      : 336752 [Complex{Float64}]
  Coordination        : 28.27947598253275
```
A few other keyword arguments are available for this particular preset (check `systempresets` in `presets.jl`). In all presets the default System keywords `dim`, `ptype` and `htype` are also available.

## Example 2: graphene flake

We now wish to build a finite-sized graphene flake of circular shape, with a radius `R=50nm`. For such kind of problem (filling a region of space with a Bravais lattice) we use the `grow` function. We first define
```julia
a0 = 0.246
incircle(r) = r[1]^2 + r[2]^2 < 50^2;
```
The system is defined by applying a model to the `:honeycomb` preset, and rescaling it to the proper lattice constant `a0`
```julia
julia> sys = System(:honeycomb, Model(Hopping(1, sublats = (1,2)))) |> transform(r-> a0 * r)
System{2,2,Float64,Complex{Float64}} : 2D system in 2D space
  Bravais vectors     : ((0.123, 0.213042), (-0.123, 0.213042))
  Sublattice names    : (:A, :B)
  Sublattice orbitals : (1, 1)
  Total sites         : 2 [Float64]
  Total hoppings      : 6 [Complex{Float64}]
  Coordination        : 3.0
```
Then we `grow` the system to fill the `incircle` region.
```julia
julia> grow(sys, region = incircle)
System{2,0,Float64,Complex{Float64}} : 0D system in 2D space
  Bravais vectors     : ()
  Sublattice names    : (:A, :B)
  Sublattice orbitals : (1, 1)
  Total sites         : 299706 [Float64]
  Total hoppings      : 897490 [Complex{Float64}]
  Coordination        : 2.9945680099831167
```
Note that the resulting lattice is `0D`, since it has zero Bravais vectors (it is bounded). 

`Elsa.jl` also has a number of region presets (see `Tuple(keys(Elsa.regionpresets))`). The above region can also be written as `Region(:circle, 50)`:
```julia
julia> System(:honeycomb, Model(Hopping(1, sublats = (1,2)))) |> transform(r-> a0 * r) |> grow(region = Region(:circle, 50))
System{2,0,Float64,Complex{Float64}} : 0D system in 2D space
  Bravais vectors     : ()
  Sublattice names    : (:A, :B)
  Sublattice orbitals : (1, 1)
  Total sites         : 299706 [Float64]
  Total hoppings      : 897490 [Complex{Float64}]
  Coordination        : 2.9945680099831167
```

By default regions start getting filled from the origin, but if the origin is not inside the region, `grow` will continue searching for a point in the region. This is the case, for example, of a ring-shaped region:
```julia
julia> ringregion(r) = 10^4 < r[1]^4 + r[2]^4 < 50^4;

julia> grow(sys, region = ringregion)
System{2,0,Float64,Complex{Float64}} : 0D system in 2D space
  Bravais vectors     : ()
  Sublattice names    : (:A, :B)
  Sublattice orbitals : (1, 1)
  Total sites         : 339640 [Float64]
  Total hoppings      : 1016788 [Complex{Float64}]
  Coordination        : 2.9937227652808858
```

## Example 3: Kane-Mele model

We now build a Kane-Mele model (http://dx.doi.org/10.1103/PhysRevLett.95.226801), which showcases a more complex use case of the `Model` API.

We start by defining a honeycomb system without any hopping (to which the Kane-Mele model will be added), the nearest and next nearest hoppings (in absolute value), and the `σ0` and `σz` Pauli matrices
```julia
sys0 = System(:honeycomb)
t1 = 1
t2 = 0.1
σ0 = @SMatrix[1 0; 0 1]
σz = @SMatrix[1 0; 0 -1]
```

We now build the Kane-Mele model in steps. First the real nearest neighbor between sublattices :A and :B of the honeycomb lattice
```julia
nearest_neighbor = Hopping(t1 * σ0, sublats = (:A, :B), range = 1)
```
By default the value of the hopping `range` is `1`, so `range` could be ommitted from the above. Also we could write `sublats` as `(1,2)`, using the sublattice index.

Note the `ndists` field, which is `missing` here (the default). When `missing`, all unit cells will be searched for neighbors from any given one, up to a distance `range`. If not `missing` only the specified inter-unit cell distance vectors (`NTuple{L,Int}`) along the `L` Bravais vectors will be considered to apply hoppings. Using this option, together with the `sublats` keyword, the next-nearest hoppings of the Kane-Mele model can be expressed as follows
```julia
next_nearest_neighbor_A = Hopping( t2 * im * σz, sublats = (:A, :A), ndists = ((1,-1), (0,1), (-1,0)))
next_nearest_neighbor_B = Hopping(-t2 * im * σz, sublats = (:B, :B), ndists = ((1,-1), (0,1), (-1,0)))
```
Note that we don't need to add the Hermitian connjugate hoppings, as they are added automatically.
We now assemble these hoppings terms into a `Model`. Several terms, as here, get summed in the final Hamiltonian.
```julia
julia> kanemele = Model(nearest_neighbor, next_nearest_neighbor_A, next_nearest_neighbor_B)
Model{Complex{Float64},3}: Model with 3 terms (0 onsite and 3 hopping)
```
Finally, we build the system by applying the above model to `sys0`
```julia
julia> sys = System(sys0, kanemele)
System{2,2,Float64,Complex{Float64}} : 2D system in 2D space
  Bravais vectors     : ((0.5, 0.866025), (-0.5, 0.866025))
  Sublattice names    : (:A, :B)
  Sublattice orbitals : (2, 2)
  Total sites         : 2 [Float64]
  Total hoppings      : 18 [Complex{Float64}]
  Coordination        : 9.0
```
The Hamiltonian of the model at a given momentum, say `k = (0.2, 0.3)` is
```
julia> Matrix(hamiltonian(sys, k = (0.2, 0.3)))
4×4 Array{Complex{Float64},2}:
 -0.00114069+0.0im              0.0+0.0im          2.92322+0.511222im          0.0+0.0im     
         0.0+0.0im       0.00114069+0.0im              0.0+0.0im           2.92322+0.511222im
     2.92322-0.511222im         0.0+0.0im       0.00114069+0.0im               0.0+0.0im     
         0.0+0.0im          2.92322-0.511222im         0.0+0.0im       -0.00114069+0.0im  
```
For clarity, we have converted from a sparse matrix (the default output of `hamiltonian`) to a dense matrix using the `Matrix` constructor.