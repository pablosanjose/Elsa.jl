#######################################################################
# System
#######################################################################

struct System{T,E,L,M<:Model,EL}
	lattice::Lattice{T,E,L,EL}
    model::M
    hamiltonian::HermitianOperator{T,L}
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
    $(sys.hamiltonian)")
end

#######################################################################
# build hamiltonian with Bloch phases
#######################################################################

function hamiltonian(sys::System{T,E,L}; k = zero(SVector{E,T}), kn = transpose(bravaismatrix(sys.lattice)) * SVector(k) / (2pi), intracell::Bool = false) where {T,E,L}
    _hamiltonian!(sys.hamiltonian, SVector{L,T}(kn), intracell)
    updatehamiltonian!(sys.hamiltonian)
    return sys.hamiltonian.matrix
end

function _hamiltonian!(hamiltonian::HermitianOperator{T,L}, kn, intracell) where {T,E,L}
    if intracell
        hamiltonian.V[hamiltonian.Voffsets[1]:end] .= zero(T)
    else
        for n in 1:(length(hamiltonian.Voffsets) - 1)
            phase = exp(1im * 2pi * dot(hamiltonian.ndist[n], kn))
            for (Vnj, Vj) in enumerate(hamiltonian.Voffsets[n]:(hamiltonian.Voffsets[n + 1] - 1))
                hamiltonian.V[Vj] = hamiltonian.Vn[n][Vnj] * phase
            end
        end
    end
    return nothing
end