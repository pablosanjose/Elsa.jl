## Type tree

The `Lattice` type contains site positions of type `T` defined in `E`-dimensional space. The lattice is periodic in `L` dimensions. `Links` between sites are included following a set of `LinkRule`

```julia
struct Lattice{T,E,L,EL}
    name::String
    bravais::Bravais{T, E, L, EL}
        matrix::SMatrix{E, L, T, EL}
    sublats::Vector{Sublat{T,E}}
        name::String
        sites::Vector{Svector{E,T}}
    links::Links{T,E,L}
        intralinks::Ilink{T,E,L}
            ndist::SVector{L, Int} # zerovector for intralinks
            slinks::Matrix{Slink{T,E}}
                targets::Vector{Int}
                srcpointers::Vector{Int}
                rdr::Vector{Tuple{SVector{E,T}, SVector{E,T}}}
                   # slink_{s1, s2} : Link goes from s1 (origin) to s2 (end)
        interlinks::Vector{Ilink{T,E,L}}
```


The `System` type bundles a `Lattice`, a `Model` and a `BlochOperator` computed from the former two.

```julia
struct System{T,E,L,EL,A}
    lattice::Lattice{T,E,L,EL}
    model::Model{OS<:Tuple, HS<:Tuple, O, H}
        onsites::OS         # Tuple of onsites
        hoppings::HS        # Tuple of hoppings
        optr::Vector{Int}   # pointers to onsites,  so that onsite(model, n) == model.onsites[optr[n]] (or default)
        hptr::Matrix{Int}   # pointers to hoppings, so that hopping(model, (n,m)) == model.hoppings[hptr[n,m]] (or default)
        dims::Vector{Int}   # site dimensions of each sublattice
        defonsite::O        # default onsite, specified with Onsite(o) instead of Onsite(n, o)
        defhopping::H       # default hopping, specified with Hopping(h) instead of Hopping((n,m), h)
        defdim::Int         # site dimension of default onsite/hopping
    h::BlochOperator{T,L}
        I::Vector{Int}              # row indices, including intercell elements
        J::Vector{Int}              # column indices, including intercell elements
        V::Vector{T}                # matrix elements, including intercell elements with zero momentum
        Voffsets::Vector{Int}       # indices in I,J,V where intercell elements begin
        Vn::Vector{Vector{T}}       # intercell matrix elements, to be copied to V with specific Bloch phases
        ndist::Vector{SVector{L, Int}}  # integer distance of different cells
        worskspace::SparseWorkspace{T}  # scratch matrices to be reused when updating h
        h::SparseMatrixCSC{Complex{T},Int}  # Preallocated BlochOperator sparse matrix h
end
```
