#######################################################################
# System
#######################################################################

struct System{T,E,L,M<:Model,EL}
	lattice::Lattice{T,E,L,EL}
    model::M
    ham::Hamiltonian{T,L}
end

System(l::Lattice, m::Model) = System(l, m, hamiltonian(l, m))

#######################################################################
# Display
#######################################################################

function Base.show(io::IO, sys::System{T,E,L}) where {T,E,L}
    print(io, "System{$T,$E,$L} : $(L)D system in $(E)D space with $T sites. 
    Bravais vectors : $(vectorsastuples(sys.lattice))
    Number of sites : $(nsites(sys.lattice))
    Sublattice names : $((sublatnames(sys.lattice)... ,))
    Unique Links : $(nlinks(sys.lattice))
    Model with sublattice site dimensions $((sys.model.dims...,)) (default $(sys.model.defdim))
    $(sys.ham)")
end

#######################################################################
# build hamiltonian with Bloch phases
#######################################################################

function hamiltonian(sys::System{T,E,L}; k = zero(SVector{E,T}), kn = transpose(sys.lattice.bravais.matrix) * SVector(k) / (2pi), intracell::Bool = false) where {T,E,L}
    _hamiltonian!(sys.ham, SVector(kn), intracell)
    updatehamiltonian!(sys.ham)
    return sys.ham.matrix
end

function _hamiltonian!(ham::Hamiltonian{T,L}, kn, intracell) where {T,E,L}
    if intracell
        ham.V[ham.Voffsets[1]:end] .= zero(T)
    else
        for n in 1:(length(ham.Voffsets) - 1)
            phase = exp(1im * 2pi * dot(ham.ndist[n], kn))
            for (Vnj, Vj) in enumerate(ham.Voffsets[n]:(ham.Voffsets[n + 1] - 1))
                ham.V[Vj] = ham.Vn[n][Vnj] * phase
            end
        end
    end
    return nothing
end