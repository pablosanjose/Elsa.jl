#######################################################################
# Kernel Polynomial Method : momenta
#######################################################################
"""
    MomentaKPM(h::AbstractMatrix; ket = missing, order = 10, randomkets = 1, bandrange = missing)

Compute the Kernel Polynomial Method (KPM) momenta `μ_n = ⟨ket|T_n(h)|ket⟩/⟨ket|ket⟩` for a 
given `ket::AbstractVector` and hamiltonian `h`, or `μ_n = Tr[T_n(h)]` if `ket` is 
`missing`, where `T_n(x)` is the Chebyshev polynomial of order `n`.

The order of the Chebyshev expansion is `order`. For the global density of states the trace 
is estimated stochastically using a number `randomkets` of random vectors. The 
`bandbrange = (ϵmin, ϵmax)` should completely encompass the full bandwidth of `hamiltonian`.
If `missing` it is computed automatically.

    MomentaKPM(system::System; kw...)

Same as above with `h = hamiltonian(sys; kw...)` (see `hamiltonian`).
"""
struct MomentaKPM{T}
    μlist::Vector{T}
    bandbracket::Tuple{T,T}
end

# CAUTION: We assume h is Hermitian here, and pass Adjoint(h) to improve mul! performance
MomentaKPM(h::AbstractMatrix; ket = missing, kw...) = _momentaKPM(h', ket; kw...)
MomentaKPM(sys::System; kw...) = MomentaKPM(hamiltonian(sys, kw...); kw...)

function _momentaKPM(h::AbstractMatrix{Tv}, ket::AbstractVector{T}; 
                     order = 10, bandrange = missing, kw...) where {T,Tv}
    μlist = zeros(real(promote_type(T, Tv)), order + 1)
    bandbracket = _bandbracket(h, bandrange)
    ket0 = normalize(ket)
    _addmomenta!(μlist, ket0, h, bandbracket)
    return MomentaKPM(jackson!(μlist), bandbracket)
end

function _momentaKPM(h::AbstractMatrix{Tv}, ket::Missing; 
                     randomkets = 1, order = 10, bandrange = missing, kw...) where {Tv}
    v = Vector{Tv}(undef, size(h, 2))
    μlist = zeros(real(Tv), order + 1)
    bandbracket = _bandbracket(h, bandrange) 
    for n in 1:randomkets
        _addmomenta!(μlist, randomize!(v), h, bandbracket)
    end
    μlist ./= randomkets
    return MomentaKPM(jackson!(μlist), bandbracket)
end

function _addmomenta!(μlist, ket, h, bandbracket)
    order = length(μlist) - 1
    ket0 = ket
    ket1 = similar(ket)
    mulscaled!(ket1, h, ket0, bandbracket)
    ket2 = similar(ket)

    μlist[1] += μ0 = 1.0
    μlist[2] += μ1 = real(ket0' * ket1)
    @showprogress for n in 3:2:(order+1)
        μlist[n] += 2 * real(ket1' * ket1) - μ0
        n + 1 > order + 1 && break
        mulscaled!(ket2, h, ket1, bandbracket)
        @. ket2 = 2 * ket2 - ket0
        μlist[n + 1] += 2 * real(ket2' * ket1) - μ1
        ket0, ket1, ket2 = ket1, ket2, ket0
    end
    return μlist
end

function mulscaled!(y, h, x, (center, halfwidth))
    mul!(y, h, x)
    @. y = (y - center * x) / halfwidth
    return y
end

function randomize!(v::AbstractVector{<:Complex})
    normalization = sqrt(length(v))
    for i in eachindex(v) 
        v[i] = exp(2 * π * 1im * rand()) / normalization
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

function _bandbracket(h, ::Missing)
    @warn "Computing spectrum bounds..."
    decompl, _ = partialschur(h, nev=1, tol=1e-4, which=LR());
    decomps, _ = partialschur(h, nev=1, tol=1e-4, which=SR());
    ϵmax = real(decompl.eigenvalues[1])
    ϵmin = real(decomps.eigenvalues[1])
    @warn  "Consider using the `bandrange` kwargs for faster performance. Computed bandrange = ($ϵmin, $ϵmax)"
    return _bandbracket(h, (ϵmin, ϵmax))
end
_bandbracket(h, (ϵmin, ϵmax), pad = 0.01) = ((ϵmax + ϵmin) / 2.0, (ϵmax - ϵmin) / (2.0 - pad))

#######################################################################
# Kernel Polynomial Method : observables
#######################################################################

"""
    dosKPM(h::AbstractMatrix; ket = missing, randomkets = 1, order = 10, resolution = 2, bandrange = missing)

Compute, using the Kernel Polynomial Method (KPM), the local density of states 
`ρ(ϵ) = ⟨ket|δ(ϵ-h)|ket⟩/⟨ket|ket⟩` for a given `ket::AbstractVector` and hamiltonian `h`, 
or the global density of states `ρ(ϵ) = Tr[δ(ϵ-h)]` if `ket` is `missing`. A tuple of energy
points `xk` and `ρ` values is returned.

The order of the Chebyshev expansion is `order`. For the global density of states the trace 
is estimated stochastically using a number `randomkets` of random vectors. The number of 
energy points `xk` is `order * resolution`, rounded to the closest integer. The 
`bandbrange = (ϵmin, ϵmax)` is computed automatically if `missing`.

    dosKPM(system::System; kw...)

Same as above with `h = hamiltonian(sys; kw...)` (see `hamiltonian`).

    dosKPM(momenta::MomentaKPM; resolution = 2)

Same as above with the KPM momenta as input (see `MomentaKPM`).
"""
dosKPM(h::AbstractMatrix; kw...) = dosKPM(MomentaKPM(h; kw...); kw...)
dosKPM(sys::System; kw...) = dosKPM(hamiltonian(sys; kw...); kw...)

function dosKPM(momenta::MomentaKPM{T}; resolution = 2, kw...) where {T}
    (center, halfwidth) = momenta.bandbracket
    numpoints = round(Int, length(momenta.μlist) * resolution)
    doslist = zeros(T, numpoints)
    copyto!(doslist, momenta.μlist)
    FFTW.r2r!(doslist, FFTW.REDFT01, 1)  # DCT-III in FFTW
    xk = [cos(π * (k + 0.5) / numpoints) for k in 0:numpoints - 1]
    @. doslist = center + halfwidth * doslist / (π * sqrt(1.0 - xk^2))
    return xk, doslist
end