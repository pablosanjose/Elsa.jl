struct KrylovKit_IRAM{Tv} <: AbstractEigenMethod{Tv}
    precond::Vector{Tv}
end

(::Type{<:KrylovKit_IRAM})(h::AbstractMatrix{Tv}) where {Tv} = KrylovKit_IRAM(rand(Tv, size(h, 2)))

function (d::Diagonalizer{<:KrylovKit_IRAM, Tv})(nev::Integer; precond = true, kw...) where {Tv}
    if isfinite(d.point)
        λs, ϕv, _ = KrylovKit.eigsolve(x -> d.lmap * x, d.method.precond, nev; kw...)
        λs .= 1 ./ λs .+ d.point
    else
        λs, ϕv, _ = KrylovKit.eigsolve(d.matrix, d.method.precond, nev, 
                                       d.point > 0 ? :LR : :SR, Lanczos(kw...))
    end
    if precond
        d.method.precond .= zero(Tv)
        foreach(ϕ -> (d.method.precond .+= ϕ), ϕv)
    end
    return Eigen(real(λs), hcat(ϕv))
end