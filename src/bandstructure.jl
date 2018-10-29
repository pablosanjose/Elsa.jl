#######################################################################
# Bandstructure
#######################################################################

struct Subband{T,N}  # N = L + 1 (nodes are energy-Blochphases)
    nodes::Lattice{T,N,0,0}
    states::Matrix{T}
    elements::Elements{T,N}
end

struct Bandstructure{T,N} 
    subbands::Vector{Subband{T,N}}
end

Bandstructure(sys::System{})