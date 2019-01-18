# Getting started

## Installation
You need Julia 1.0 or higher to use `Elsa.jl`. Until I register the package it should be installed as follows
```
julia>] add https://github.com/pablosanjose/Elsa.jl 
```
Then, to start using it in a Julia session do `using Elsa`

Optionally, you can also install `ElsaPlots.jl` to visualise lattices, bandstructures, etc.
```
julia>] add https://github.com/pablosanjose/ElsaPlots.jl 
```

## Basic concepts

There is a hierarchy of types in `Elsa.jl` that should be understood in order to get started. In bold are the most important ones.

- `Sublat{T,E}`: a set of sites, in the form of positions `SVector{E,T}` (`E` is the embedding dimension, and `T<:Number` is the type of coordinates)
    - We never *need* to enter `SVector`s when using `Elsa.jl`. Tuples will do. 
    - *Example*: `Sublat((1.0,0), (0,1))` will promote the two site positions to `SVector{Float64,2}`

- `Bravais`: a set of Bravais vectors of a lattice
    - *Example*: `Bravais((1.0,0), (0,1))` are the Bravais vectors of a square lattice in 2D.

- `Links`: a set of directives that establish which sites in a set of `Sublat`s are linked. The used will control links indirectly, using `LinkRule`s.

- **`Lattice{T,E,L}`**: A collection of `Sublat`s, optionally bundled with a `Bravais` and some `Links`.
    - It can represent an infinite lattice, or a bounded lattice. The lattice dimension `L` denotes the number of unbounded axes.
    - To build a Lattice we can use built-in `Preset`s or chain `LatticeDirective`s in any given order
- `LatticeDirective`s: abstract type whose subtypes can be used to build lattices. `LatticeDirective` subtypes include `Sublat`, `Bravais`, `LinkRule`, `Region`, `Supercell`, `LatticeConstant`, `Dim` and `Precision`.

- **`Model`**: A set of `Onsite` and `Hopping` directives that specify the Hamiltonian matrices on a site at a given position in a given `Sublat` and between two given positions and `Sublat`s, respectively. These can be numbers or matrices themselves (for sites with more than one orbital).

- **`System{T,E,L}`**: a combination of a `Lattice` and a `Model`. 
    - Upon construction, a `System` computes and stores the Bloch harmonics needed to compute the Bloch Hamiltonian `H(k)` of the system at momentum `k` efficiently (`SparseMatrixCSC` format).

- `Bandstructure`: a bandstructure of a `System`, understood as the eigenvalues and eigenvectors as a function of Bloch momenta. Connectivity within subbands is computed automatically from eigenstates.

Other work-in-progress types include `GreenFunction`, `Perturbation`, `CompositeSystem`, etc. Apart from types and their constructors, some utility functions are also available, such as `lattice!`, `transform`, `combine`, `wrap`, `mergesublats`, `velocity`, etc.

## Simple example, step by step
Let us build an infinite graphene lattice, linked to nearest neighbors. For the sake of clarity I describe below the fully manual way, which is highly verbose and wouldn't be the way to do things in practice.

The general idea is that a lattice is built in steps, by supplying a set of `LatticeDirective`s to the `Lattice` constructor, that are implemented sequentially. In this example we use three `LatticeDirectives`, namely `Sublat`, `Bravais` and `LinkRule`. 
- Define the lattice constant in nanometers
```julia
a0 = 0.246;
```
- Create a sublattice of type `Sublat <: LatticeDirective` that contains a single site at position `(0.0, -0.5/sqrt(3.0) * a0)`, and name it sublattice `:A`
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
- Create a `LinkRule <: LatticeDirective` to link all sites within a given distance `a0/√3`
```julia
lr = LinkRule(a0/√3);
```
- Pass all these directives to the `Lattice` constructor. The order usually matters, though not in this particular case
```julia
julia> lat = Lattice(subA, subB, br, lr)
Lattice{Float64,2,2} : 2D lattice in 2D space with Float64 sites
    Bravais vectors  : ((0.123, 0.213042), (-0.123, 0.213042))
    Sublattice names : (:A, :B)
    Total sites      : 2
    Total links      : 5
    Coordination     : 3.0
```
The resulting object is of type `Lattice{T,E,L}`, where `T = Float64` is the numeric type of positions, `E = 2` is the embedding dimension (2D space in this example), and `L = 2` is the lattice dimension (two because graphene is a 2D Bravais lattice, with two bravais vectors). The reported coordination is the average number of links out of each site (3 in graphene with nearest neighbors). The number `Total links` indicates how many links exist between sites belonging to a given unit cell plus the links of those sites and sites in neighboring unit cells. In this case there are five, one intra- and four intercell:
```
 \/
 |
 /\
```

### The order of `LatticeDirective`s matters

While the above is a perfectly viable way to build a lattice, some advanced users might prefer to use single-liners for such simple lattices:
```julia
julia> using Elsa

julia> Lattice(Sublat((0.0, -0.5/sqrt(3.0)); name = :A), 
               Sublat((0.0,  0.5/sqrt(3.0)); name = :B), 
               Bravais(( cos(pi/3), sin(pi/3)), 
                       (-cos(pi/3), sin(pi/3))), 
               LinkRule(1/√3), 
               LatticeConstant(0.246))
Lattice{Float64,2,2} : 2D lattice in 2D space with Float64 sites
    Bravais vectors  : ((0.5, 0.866025), (-0.5, 0.866025))
    Sublattice names : (:A, :B)
    Total sites      : 2
    Total links      : 5
    Coordination     : 3.0
```
In this case we did not define `a0`, but rather assumed it to be 1. We added a final directive `LatticeConstant(0.246)` whose role is to rescale the lattice uniformly (including its Bravais vectors) so that it has the specified lattice constant. In this example we see an important property of the Elsa.jl API for bulding lattices: the order of directives matters. If we switch the last two directives for example we get
```julia
julia> Lattice(Sublat((0.0, -0.5/sqrt(3.0)); name = :A), 
                      Sublat((0.0,  0.5/sqrt(3.0)); name = :B), 
                      Bravais(( cos(pi/3), sin(pi/3)), 
                              (-cos(pi/3), sin(pi/3))),
                      LatticeConstant(0.246), 
                      LinkRule(1/√3))
Lattice{Float64,2,2} : 2D lattice in 2D space with Float64 sites
    Bravais vectors  : ((0.123, 0.213042), (-0.123, 0.213042))
    Sublattice names : (:A, :B)
    Total sites      : 2
    Total links      : 77
    Coordination     : 39.0
```
The number of links has now increased: the `LinkRule(1/√3)` was applied *after* the lattice rescaling, so that many more sites become within the `1/√3` range as before.

### Presets

The above can be streamlined further by using built-in presets. currently available Lattice presets are:
```julia
julia> Tuple(keys(Elsa.latticepresets))
(:bcc, :graphene, :honeycomb, :cubic, :linear, :fcc, :honeycomb_bilayer, :square, :triangular)
```

We can use a preset using a special directive `Preset`
```julia
julia> Lattice(Preset(:graphene))
Lattice{Float64,2,2} : 2D lattice in 2D space with Float64 sites
    Bravais vectors  : ((0.123, 0.213042), (-0.123, 0.213042))
    Sublattice names : (:A, :B)
    Total sites      : 2
    Total links      : 5
    Coordination     : 3.0
```
or more simply `Lattice(:graphene)`. Some more complex presets can take optional keyword arguments, for example 
```julia
julia> Lattice(Preset(:honeycomb_bilayer, twistindex = 31))
Lattice{Float64,3,2} : 2D lattice in 3D space with Float64 sites
    Bravais vectors  : ((15.5, 26.846788, 0.0), (-15.5, 26.846788, 0.0))
    Sublattice names : (:Ab, :Bb, :At, :Bt)
    Total sites      : 3844
    Total links      : 5890
    Coordination     : 3.0
```
for a twisted honeycomb bilayer with moiré index `31` (corresponding to the so-called first magic angle).

We can also combine presets and `LatticeDirective`s, by specifying the directives *after* the preset
```julia
Lattice(:graphene, Sublat((0, 0), name = :C), LatticeConstant(0.311), LinkRule(0.5), Dim(3), Precision(Float32))
Lattice{Float32,3,2} : 2D lattice in 3D space with Float32 sites
    Bravais vectors  : ((0.1555f0, 0.269334f0, 0.0f0), (-0.1555f0, 0.269334f0, 0.0f0))
    Sublattice names : (:A, :B, :C)
    Total sites      : 3
    Total links      : 83
    Coordination     : 28.666666666666668
```
Again, `LatticeDirectives` are applied in order on top of the preset, from left to right, incrementally modifying the lattice. In the example above we have: (1) started from graphene, (2) added a new sublattice with a single site at the origin, (3) rescaled the whole lattice to have a `0.311` lattice constant, (4) changed its linking radius to 0.5, (5) promoted the embedding space to 3D space, and (6) changed all positions to use `Float32` instead of the default `Float64`.

## A slightly more advanced example

We now wish to build a finite-sized graphene flake of circular shape, with a radius `R=50nm`. For such kind of problem (filling a region of space with a Bravais lattice) we use the `LatticeDirective` `Region`. 

A `Region` takes a boolean function that specifies whether a point belongs to a region, and optionally a kwarg `seed` that specifies a point where the region begins to be filled by a Bravais lattice. for example we could do
```
incircle(r) = r[1]^2 + r[2]^2 < 50^2
Lattice(:graphene, Region{2}(incircle))
``` 
Note the `{2}` in the `Region{E}(::Function)` constructor, which specifies the embedding dimensions `E` of the region (it cannot be inferred from the function itself). `Region` also accepts a number of presets (see `Tuple(keys(Elsa.regionpresets))`). The above region can also be written as `Region(:circle, 50)`:
```julia
julia> lat = Lattice(:graphene, Region(:circle, 50))
Lattice{Float64,2,0} : 0D lattice in 2D space with Float64 sites
    Bravais vectors  : ()
    Sublattice names : (:A, :B)
    Total sites      : 299706
    Total links      : 448745
    Coordination     : 2.9945680099831167
```

By default regions start getting filled from the origin. If a region does not include the origin, a `seed` to start filling must be specified. This is the case, for example, of a ring-shaped region:
```julia
julia> myregion(r) = 10^4 < r[1]^4 + r[2]^4 < 50^4;

julia> Lattice(:graphene, Region{2}(myregion, seed = (0,20)))
Lattice{Float64,2,0} : 0D lattice in 2D space with Float64 sites
    Bravais vectors  : ()
    Sublattice names : (:A, :B)
    Total sites      : 339626
    Total links      : 508352
    Coordination     : 2.993598841078127
```


### Building a system

We want a single orbital per site. Let us define the hopping between each linked site as `1`, and make the onsite energy a random Anderson disorder between `-0.1` and `0.1`
```julia
julia> lat = Lattice(:graphene, Region(:circle, 50));

julia> sys = System(lat, Model(Hopping(1), Onsite(r -> 0.2*(rand() - 0.5))))
System{Float64,2,0} : 0D system in 2D space with Float64 sites.
    Bravais vectors : ()
    Number of sites : 299706
    Sublattice names : (:A, :B)
    Unique Links : 448745
    Model with sublattice site dimensions () (default 1)
    Bloch operator of dimensions (299706, 299706) with 1197196 elements
```
The line `Model...` specifies Model defaults (not covered in this quick-start guide). The last line specifies the dimensions of the sparse Hamiltonian of the system (intra and intercell) and its number of non-zero elements. 

If we want more orbitals we could enter a matrix, as in `Onsite([1 2; 2 3])` or even specify different onsite/hoppings for different `Sublat`s. For example this adds disorder only to sublattice 2
```julia
julia> sys = System(lat, Model(Hopping(1), Onsite(r -> 0.2*(rand() - 0.5), 2)))
System{Float64,2,0} : 0D system in 2D space with Float64 sites.
    Bravais vectors : ()
    Number of sites : 299706
    Sublattice names : (:A, :B)
    Unique Links : 448745
    Model with sublattice site dimensions (1, 1) (default 1)
    Bloch operator of dimensions (299706, 299706) with 1047343 elements
```

### Visualising the result
`ElsaPlots.jl` can be used to visualise a number of `Elsa.jl` objects in 3D using `Makie.jl`. For example

```julia
julia> using ElsaPlots

julia> plot(sys)
```
(This is currently broken due to rapid changes in the `Makie` package that `ElsaPlots` uses.)

### Extracting the Hamiltonian

We can obtain the Hamiltonian of the closed system as `hamiltonian(sys)`, which returns a `SparseMatrixCSC`.

For an unbounded lattice (dimension `L !=0 `) we specify Bloch momenta as `hamiltonian(sys, k = (kx, ky))` or Bloch phases `ϕᵢ=k⋅aᵢ` as `hamiltonian(sys, kn = (ϕ1, ϕ2))`.