# struct Arpack_IRAM{Tv} <: AbstractEigenMethod{Tv}
#     precond::Vector{Tv}
# end

# (::Type{<:Arpack_IRAM})(h::AbstractMatrix{Tv}) where {Tv} = Arpack_IRAM(rand(Tv, size(h, 2)))

# function (d::Diagonalizer{<:Arpack_IRAM,Tv})(nev::Integer; precond = true, kw...) where {Tv}
#     if isfinite(d.point)
#         which = :LM
#         # sigma = real(Tv) === Tv ? d.point : d.point + 1.0im
#         sigma = d.point
#     elseif point > 0
#         which = :LR
#         sigma = nothing
#     else
#         which = :SR
#         sigma = nothing
#     end
#     λs, ϕs, _ = Arpack.eigs(d.matrix; nev = nev, sigma = sigma, which = which,
#                              v0 = d.method.precond, kw...)
#     if precond
#         d.method.precond .= zero(Tv)
#         foreach(ϕ -> (d.method.precond .+= ϕ), eachcol(ϕs))
#     end
#     return Eigen(real(λs), ϕs)
# end

ArpackPackage(; sigma = 1im, kw...) = ArpackPackage((sigma = sigma, values(kw)...))