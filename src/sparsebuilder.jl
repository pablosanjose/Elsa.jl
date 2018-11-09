mutable struct SparseMatrixBuilder{T}
    m::Int
    n::Int
    colptr::Vector{Int}
    rowval::Vector{Int}
    nzval::Vector{T}
    colcounter::Int
    rowvalcounter::Int
end

function SparseMatrixBuilder(::Type{T}, m, n, coordinationhint = 1) where T
    colptr = Vector{Int}(undef, n + 1)
    colptr[1] = 1
    rowval = Int[]; sizehint!(rowval, round(Int, 0.5 * coordinationhint * n))
    nzval = T[];    sizehint!(nzval,  round(Int, 0.5 * coordinationhint * n))
    # The 0.5 is due to storing undirected links only
    return SparseMatrixBuilder(m, n, colptr, rowval, nzval, 1, 1)
end

function pushtocolumn!(s::SparseMatrixBuilder, row, x, skipdupcheck = true)
    if skipdupcheck || !isintail(row, s.rowval, s.colptr[s.colcounter])
        push!(s.rowval, row)
        push!(s.nzval, x)
        s.rowvalcounter += 1
    end
    return x
end

function isintail(element, container, start::Int) 
    for i in start:length(container)
        container[i] == element && return true
    end
    return false
end

function finalisecolumn!(s::SparseMatrixBuilder)
    s.colcounter > s.n && throw(DimensionMismatch("Pushed too many columns to matrix"))
    s.colcounter += 1
    s.colptr[s.colcounter] = s.rowvalcounter
    return nothing
end

function SparseArrays.sparse(s::SparseMatrixBuilder)
    if s.colcounter < s.n + 1
        for col in (s.colcounter + 1):(s.n + 1) 
            s.colptr[col] = s.rowvalcounter
        end
    end
    return SparseMatrixCSC(s.m, s.n, s.colptr, s.rowval, s.nzval)
end