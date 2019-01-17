# Getting started

## Installation
You need Julia 1.0 or higher to use `Elsa.jl`. Until I register the package it should be installed as follows
```
julia>] add https://github.com/pablosanjose/Elsa.jl 
```
Then, to start using it in a Julia session do `using Elsa`

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
- `LatticeDirective`s: abstract type whose subtypes can be used to build lattices and include `Sublat`, `Bravais`, `LinkRule`, `Region`, `Supercell`, `LatticeConstant`, `Dim` and `Precision`.

- **`Model`**: A set of `Onsite` and `Hopping` directives that specify the Hamiltonian matrices on a site at a given position in a given `Sublat` and between two given positions and `Sublat`s, respectively. These can be numbers or matrices themselves (for sites with more than one orbital).

- **`System{T,E,L}`**: a combination of a `Lattice` and a `Model`. 
    - Upon construction, a `System` computes and stores the Bloch harmonics needed to compute the Bloch Hamiltonian `H(k)` of the system at momentum `k` efficiently (`SparseMatrixCSC` format).

## Simple example
Let us build a circular graphene flake of radius , linked to nearest neighbors, and with 

### Building a lattice
We start by building a lattice with links between any two sites at distance smaller or equal to `1/√3`
```julia
julia> using Elsa

julia> Lattice(Sublat((0.0, -0.5/sqrt(3.0)); name = :A), Sublat((0.0, 0.5/sqrt(3.0)); name = :B), Bravais((cos(pi/3), sin(pi/3)), (-cos(pi/3), sin(pi/3))), LinkRule(1/√3))
Lattice{Float64,2,2} : 2D lattice in 2D space with Float64 sites
    Bravais vectors  : ((0.5, 0.866025), (-0.5, 0.866025))
    Sublattice names : (:A, :B)
    Total sites      : 2
    Total links      : 5
    Coordination     : 3.0
```
The same can be achieved using presets, either `Lattice(:honeycomb)` which has no links, just sites, or `Lattice(:graphene)` which has nearest neighbor links as above. Note that by default presets produce lattices with lattice constant 1 (can be overriden with the  `LatticeConstant`, a `LatticeDirective`)

We can also combine presets and `LatticeDirective`s
```
julia> Lattice(:honeycomb, LinkRule(2))
Lattice{Float64,2,2} : 2D lattice in 2D space with Float64 sites
    Bravais vectors  : ((0.5, 0.866025), (-0.5, 0.866025))
    Sublattice names : (:A, :B)
    Total sites      : 2
    Total links      : 59
    Coordination     : 30.0
```
Note that `LatticeDirective`s further to the right override the preceding ones if necessary. 

To build a circular disk of radius `R=100` in units of the lattice constant `a0` (here `1`) we do
```
julia> lat = Lattice(:graphene, Region(:circle, 100))
Lattice{Float64,2,0} : 0D lattice in 2D space with Float64 sites
    Bravais vectors  : ()
    Sublattice names : (:A, :B)
    Total sites      : 72562
    Total links      : 108445
    Coordination     : 2.989030070835975
```
The directive `Region(:circle, 100)` is a region preset. Any other region can be built with a boolean function  that specifies whether a point at `r` is in the region (it should be bounded). For performance reasons it is better to define it before calling `Region`. It is also necessary to specify the embedding dimension `E`, using the syntax `Region{E}(::Function)`
```
julia> myregion(r) = r[1]^4 + r[2]^4 < 100^4;

julia> Lattice(:graphene, Region{2}(myregion))
Lattice{Float64,2,0} : 0D lattice in 2D space with Float64 sites
    Bravais vectors  : ()
    Sublattice names : (:A, :B)
    Total sites      : 85574
    Total links      : 127889
    Coordination     : 2.9889686119615773
```

### Building a system

We want a single orbital per site. Let us define the hopping between each linked site as `1`, and make the onsite energy a random Anderson disorder between `-0.1` and `0.1`
```
julia> sys = System(lat, Model(Hopping(1), Onsite(r -> 0.2*(rand() - 0.5))))
```
If we want more orbitals we could enter a matrix, as in `Onsite([1 2; 2 3])` or even specify different onsite/hoppings for different `Sublat`s. For example this adds disorder only to sublattice 2
```
julia> sys = System(lat, Model(Hopping(1), Onsite(r -> 0.2*(rand() - 0.5), 2)))
```

## Extracting the Hamiltonian

We can obtain the Hamiltonian of the closed system as `hamiltonian(sys)`.

For an unbounded lattice (dimension `L !=0 `) we specify Bloch momenta as `hamiltonian(sys, k = (kx, ky))` or Bloch phases `ϕᵢ=k⋅aᵢ` as `hamiltonian(sys, k = (ϕ1, ϕ2))`.