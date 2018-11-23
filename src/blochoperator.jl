#######################################################################
# BlochOperator
#######################################################################

struct SparseWorkspace{T}
    klasttouch::Vector{Int}
    csrrowptr::Vector{Int}
    csrcolval::Vector{Int}
    csrnzval::Vector{Complex{T}}
end
function SparseWorkspace{T}(dimh, lengthI) where {T}
    klasttouch = Vector{Int}(undef, dimh)
    csrrowptr = Vector{Int}(undef, dimh + 1)
    csrcolval = Vector{Int}(undef, lengthI)
    csrnzval = Vector{Complex{T}}(undef, lengthI)
    SparseWorkspace{T}(klasttouch, csrrowptr, csrcolval, csrnzval)
end

struct BlochOperator{T,L}
    I::Vector{Int}
    J::Vector{Int}
    V::Vector{Complex{T}}
    Voffsets::Vector{Int}			# offset of Vn in V
    Vn::Vector{Vector{Complex{T}}}  # one per ndist
    ndist::Vector{SVector{L, Int}}
    workspace::SparseWorkspace{T}
    matrix::SparseMatrixCSC{Complex{T},Int}
end

function Base.show(io::IO, op::BlochOperator)
    print(io, "Bloch operator of dimensions $(size(op.matrix)) with $(nnz(op.matrix)) elements")
end

function insertblochphases!(bop::BlochOperator{T,L}, kn::SVector, intracell) where {T,L}
	L >= 0 && _insertblochphases!(bop.V, bop.Vn, bop.Voffsets, bop.ndist, convert(SVector{L,T}, kn), intracell)
	return nothing
end

#######################################################################
# BlochVector
#######################################################################

struct BlochVector{T,L}
    I::Vector{Int}
    J::Vector{Int}
    V::Vector{Complex{T}}
    Voffsets::Vector{Int}
    Vns::NTuple{L, Vector{Vector{Complex{T}}}}
    ndist::Vector{SVector{L, Int}}
    workspace::SparseWorkspace{T}
    matrix::SparseMatrixCSC{Complex{T},Int}
end

function Base.show(io::IO, op::BlochVector{T,L}) where {T,L}
    print(io, "Bloch $L-vector of dimensions $(size(op.matrix)) with $(nnz(op.matrix)) elements")
end

function insertblochphases!(bvec::BlochVector{T,L}, kn, axis) where {T,L}
	L > 0 && _insertblochphases!(bvec.V, bvec.Vns[axis], bvec.Voffsets, bvec.ndist, convert(SVector{L,T}, kn), false)
	return nothing
end

function gradient(op::BlochOperator{T,L}; kn::SVector{L,Int} = zero(SVector{L,Int}), axis::Int = 1) where {T,L}
    ndist = op.ndist
	workspace = op.workspace
	dim = size(op.matrix, 1)
	offset = op.Voffsets[1]
	I = op.I[offset:end]
	J = op.J[offset:end]
	V = zeros(Complex{T}, length(I))
	Voffsets = op.Voffsets .- offset .+ 1
	Vns = ntuple(ax -> [(2pi * im * n[ax]) .* v for (n, v) in zip(ndist, op.Vn)], Val(L))
	L > 0 && _insertblochphases!(V, Vns[axis], Voffsets, ndist, kn, false)
	matrix = sparse!(I, J, V, dim, workspace)
	return BlochVector(I, J, V, Voffsets, Vns, ndist, workspace, matrix)
end

function _insertblochphases!(V, Vn, Voffsets, ndist, kn, intracell)
    if intracell
        V[Voffsets[1]:end] .= zero(T)
    else
        for n in 1:(length(Voffsets) - 1)
            phase = exp(2pi * im * dot(ndist[n], kn))
            for (Vnj, Vj) in enumerate(Voffsets[n]:(Voffsets[n + 1] - 1))
                V[Vj] = Vn[n][Vnj] * phase
            end
        end
    end
    return nothing
end
