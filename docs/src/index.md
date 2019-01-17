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

## Simple example
Let us build a circular graphene flake of radius `R=100 a0`, linked to nearest neighbors, and with some random Anderson disorder. Our unit of distance will be nanometers, so that the lattice constant is `a0=0.246`.

### Building a lattice
We start by building a lattice with links between any two sites at distance smaller or equal to `1/√3a0`. The fully manual approach would be
```julia
julia> using Elsa

julia> Lattice(Sublat((0.0, -0.5/sqrt(3.0)); name = :A), Sublat((0.0, 0.5/sqrt(3.0)); name = :B), Bravais((cos(pi/3), sin(pi/3)), (-cos(pi/3), sin(pi/3))), LinkRule(1/√3), LatticeConstant(0.246))
Lattice{Float64,2,2} : 2D lattice in 2D space with Float64 sites
    Bravais vectors  : ((0.5, 0.866025), (-0.5, 0.866025))
    Sublattice names : (:A, :B)
    Total sites      : 2
    Total links      : 5
    Coordination     : 3.0
```
Note that `LatticeDirective`s are applied in order, from left to right. Order is important. In the case above, `LatticeConstant(0.246)` is applied only *after* defining all previous directives. If we apply `LinkRule` after `LatticeConstant` we would need `LinkRule(0.246/√3)` to get the same result as above.

The above can be achieved more simply using presets, either `Lattice(Preset(:honeycomb))` (or simply `Lattice(:honeycomb)`) which has no links and `a0 = 1`, or alternatively `Lattice(Preset(:graphene))` (or simply `Lattice(:graphene)`) which has nearest-neighbor links as above, and the proper graphene `a0=0.246` lattice constant. 

We can also combine presets and `LatticeDirective`s
```julia
julia> Lattice(:honeycomb, Sublat((0, 0), name = :C), LinkRule(2), LatticeConstant(0.311), Dim(3), Precision(Float32))
Lattice{Float32,3,2} : 2D lattice in 3D space with Float32 sites
    Bravais vectors  : ((0.1555f0, 0.269334f0, 0.0f0), (-0.1555f0, 0.269334f0, 0.0f0))
    Sublattice names : (:A, :B, :C)
    Total sites      : 3
    Total links      : 131
    Coordination     : 44.666666666666664
```
More complex presets can take keyword arguments, e.g. `Lattice(Preset(:honeycomb_bilayer, twistindex = 31))` for a magic-angle twisted honeycomb bilayer.

To build a circular graphene disk of radius `R=50nm` we do
```julia
julia> lat = Lattice(:graphene, Region(:circle, 50))
Lattice{Float64,2,0} : 0D lattice in 2D space with Float64 sites
    Bravais vectors  : ()
    Sublattice names : (:A, :B)
    Total sites      : 299706
    Total links      : 448745
    Coordination     : 2.9945680099831167
```
The directive `Region(:circle, 50)` is a region preset. Any other region can be built with a boolean function  that specifies whether a point at `r` is in the region (it should be bounded). For performance reasons it is better to define it before calling `Region`. It is also necessary to specify the embedding dimension `E`, using the syntax `Region{E}(::Function)`
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
The optional `seed` specifies at which point in space to begin filling the region.

### Building a system

We want a single orbital per site. Let us define the hopping between each linked site as `1`, and make the onsite energy a random Anderson disorder between `-0.1` and `0.1`
```julia
julia> sys = System(lat, Model(Hopping(1), Onsite(r -> 0.2*(rand() - 0.5))))
```
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