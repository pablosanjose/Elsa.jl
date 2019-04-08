# function IDCT(realvector, numnodes, bandwidth, bandcenter)
#     FFTW.r2r!(realvector, FFTW.REDFT01, 1)     #DCT of type III
#     nodes = [cos(π * (k + 0.5) / numnodes) for k in 0:numnodes-1]
#     @. realvector = bandcenter + realvector / (bandwidth * (π*sqrt(1.0-nodes^2)))
#     return nodes, realvector
# end

function momenta!(ket::AbstractVector{T}, h::AbstractMatrix{Tv}; order = 10, kw...) where {T,Tv}
    momenta = zeros(real(promote_type(T, Tv)), order + 1)    
    _addmomenta!(momenta, ket, h; kw...)
    return jackson!(momenta)
end

function momenta!(h::AbstractMatrix{Tv}; randomvectors = 1, order = 10, kw...) where {Tv}
    v = Vector{Tv}(undef, size(h, 2))
    momenta = zeros(real(Tv), order + 1)
    for n in 1:randomvectors
        _addmomenta!(momenta, randomize!(v), h; kw...)
    end
    momenta ./= randomvectors
    return jackson!(momenta)
end

function _addmomenta!(momenta, ket, h; bandwidth = 1.0, bandcenter = 0.0)
    order = length(momenta) - 1
    ket0 = ket
    ket1 = similar(ket)
    mulscaled!(ket1, h, ket0, bandwidth, bandcenter)
    ket2 = similar(ket)

    momenta[1] += μ0 = 1.0
    momenta[2] += μ1 = real(ket0' * ket1)
    for n in 3:2:(order+1)
        momenta[n] += 2 * real(ket1' * ket1) - μ0
        n + 1 > order + 1 && break
        mulscaled!(ket2, h, ket1, bandwidth, bandcenter)
        @. ket2 = 2 * ket2 - ket0
        momenta[n + 1] += 2 * real(ket2' * ket1) - μ1
        ket0, ket1, ket2 = ket1, ket2, ket0
    end
    return momenta
end

function mulscaled!(y, h, x, bandwidth, bandcenter) 
    mul!(y, h, x)
    @. y = (y - bandcenter * x) / bandwidth
    return y
end

function randomize!(v::AbstractVector{<:Complex})
    n = length(v)
    for i in eachindex(v) 
        v[i] = exp(2 * π * 1im * rand()) / n
    end
    return v
end

function randomize!(v::AbstractVector{<:Real})
    for i in eachindex(v) 
        v[i] = randn()
    end
    normalize!(v)
    return v
end

function jackson!(μ::AbstractVector) 
    order = length(μ) - 1
    for n in eachindex(μ)
        μ[n] *= ((order - n + 1) * cos(π * n / (order + 1)) + 
                sin(π * n / (order + 1)) * cot(π / (order + 1))) / (order + 1)
    end
    return μ
end