#######################################################################
# Kernel Polynomial Method : momenta
#######################################################################
struct MomentaKPM{T}
    μlist::Vector{T}
    bandbracket::Tuple{T,T}
end

"""
    momentaKPM(h::AbstractMatrix; ket = missing, obs = missing, order = 10, randomkets = 1, bandrange = missing)

Compute the Kernel Polynomial Method (KPM) momenta `μ_n = ⟨ket|A T_n(h)|ket⟩/⟨ket|ket⟩` where `T_n(x)` 
is the Chebyshev polynomial of order `n`, for a given `ket::AbstractVector`, hamiltonian `h`, and 
observable `A`. If `ket` is `missing` computes the momenta by means of a stochastic trace
`μ_n = Tr[A T_n(h)]`. If `ket::AbstractMatrix` the trace is computed over a user-specified set of kets stored
in `ket`. In the last case, the kets are encoded as the columns (or rows) of a size(h) diagonal sparse matrix.
`A = I` if `obs` is `missing`.

The order of the Chebyshev expansion is `order`. For the global density of states the trace
is estimated stochastically using a number `randomkets` of random vectors. The
`bandbrange = (ϵmin, ϵmax)` should completely encompass the full bandwidth of `hamiltonian`.
If `missing` it is computed automatically.

# Example
```
julia> h = LatticePresets.cubic() |> hamiltonian(hopping(1)) |> unitcell(region = RegionPresets.sphere(10));

julia> Elsa.MomentaKPM(bloch(h), bandrange = (-6,6))
Elsa.MomentaKPM{Float64}([0.9594929736144973, -0.005881595972403821, -0.4933354572913581, 0.00359537502632597, 0.09759451291347333, -0.0008081453185250322, -0.00896262538765363, 0.00048205637037715177, -0.0003705198310034668, 9.64901673962623e-20, 9.110915988898614e-18], (0.0, 6.030150753768845))
```
"""
momentaKPM(h::AbstractMatrix; ket = missing, obs = missing, kw...) = _momentaKPM(h, ket, obs; kw...)

function _momentaKPM(h::AbstractMatrix{Tv}, ket::AbstractVector{T}, obs; order = 10, bandrange = missing, kw...) where {T,Tv}
   
    μlist = zeros(real(promote_type(T, Tv)), order + 1)
    bandbracket = _bandbracket(h, bandrange)
    ket0 = normalize(ket)
    _addmomenta!(μlist, ket0, h, bandbracket, obs)
    return MomentaKPM(jackson!(μlist), bandbracket)
end

function _momentaKPM(h::AbstractMatrix{Tv}, ket::AbstractMatrix{T}, obs; order = 10, bandrange = missing, kw...) where {T,Tv}
    v = zeros(Tv, size(h,1))
    μlist = zeros(real(Tv), order + 1)
    bandbracket = _bandbracket(h, bandrange)
    numkets = Int64(sum(nonzeros(ket)))
    pos = [i[1] for i in findall(!iszero, ket)]; 
    for n in pos
        v[n] = 1.0
        _addmomenta!(μlist, v, h, bandbracket, obs)
        v[n] = 0.0
    end
    μlist ./= numkets
    return MomentaKPM(jackson!(μlist), bandbracket)
end

function _momentaKPM(h::AbstractMatrix{Tv}, ket::Missing, obs; randomkets = 1, order = 10, bandrange = missing, kw...) where {Tv}
    v = Vector{Tv}(undef, size(h, 2))
    μlist = zeros(real(Tv), order + 1)
    bandbracket = _bandbracket(h, bandrange)
    for n in 1:randomkets
        _addmomenta!(μlist, randomize!(v), h, bandbracket, obs)
    end
    μlist ./= randomkets
    return MomentaKPM(jackson!(μlist), bandbracket)
end

function _addmomenta!(μlist, ket, h, bandbracket, obs::Missing)
    order = length(μlist) - 1
    ket0 = ket
    ket1 = similar(ket)
    mulscaled!(ket1, h, ket0, bandbracket)
    ket2 = similar(ket)
    μlist[1]  += μ0 = 1.0
    μlist[2]  += μ1 = real(ket0' * ket1)
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

function _addmomenta!(μlist, ket, h, bandbracket, obs::AbstractMatrix)
    order = length(μlist) - 1
    ket0 = ket
    ketL = similar(ket)
    ket1 = similar(ket)
    ket2 = similar(ket)
    mul!(ketL', ket', obs) 
    mulscaled!(ket1, h, ket0, bandbracket)
    μlist[1] = real(ketL' * ket0)
    μlist[2] = real(ketL' * ket1)
    @showprogress for n in 3:1:(order+1) 
        mulscaled!(ket2, h, ket1, bandbracket)
        @. ket2 = 2 * ket2 - ket0
        μlist[n] += real(ketL' * ket2)
        n + 1 > order + 1 && break
        ket0, ket1, ket2 =  ket1, ket2, ket0 
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

function keti!(v, dim, i)
    v = zeros(v)
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
    @warn "Computing spectrum bounds... Consider using the `bandrange` kwargs for faster performance."
    checkloaded(:ArnoldiMethod)
    decompl, _ = Main.ArnoldiMethod.partialschur(h, nev=1, tol=1e-4, which = Main.ArnoldiMethod.LR());
    decomps, _ = Main.ArnoldiMethod.partialschur(h, nev=1, tol=1e-4, which = Main.ArnoldiMethod.SR());
    ϵmax = real(decompl.eigenvalues[1])
    ϵmin = real(decomps.eigenvalues[1])
    @warn  "Computed bandrange = ($ϵmin, $ϵmax)"
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
 """
#    dosKPM(momenta::MomentaKPM; resolution = 2)
#Same as above with the KPM momenta as input (see `MomentaKPM`).

dosKPM(h::AbstractMatrix; obs = missing, kw...) = dosKPM(momentaKPM(h, obs = missing; kw...); kw...) 

function dosKPM(momenta::MomentaKPM{T}; resolution = 2, kw...) where {T}
    (center, halfwidth) = momenta.bandbracket
    numpoints = round(Int, length(momenta.μlist) * resolution)
    doslist = zeros(T, numpoints)
    copyto!(doslist, momenta.μlist)
    FFTW.r2r!(doslist, FFTW.REDFT01, 1)  # DCT-III in FFTW
    xk = [cos(π * (k + 0.5) / numpoints) for k in 0:numpoints - 1]
    @. doslist = center + halfwidth * doslist / (π * sqrt(1.0 - xk^2))
    @. xk = center + halfwidth * xk
    return xk, doslist
end

"""
thermalaverageKPM(A::AbstractMatrix, h::AbstractMatrix; ketset = missing, T = 0 ,ket = missing, randomkets = 1, order = 10, resolution = 2, bandrange = missing)

Compute, using the Kernel Polynomial Method (KPM), the thermal expectation value
`<A> = Σ_k f(E_k,E_F) <k|A|k> =  ∫dE f(E,E_f) Tr [A δ(E-H)] = Tr [A f(H,E_f)]`
for a given hermitian operator `A` and a hamiltonian `h`. If `ketset::AbstractMatrix` or
`ketset::AbstractMatrix` computes the thermal average over the kets in ketset (see momentaKPM).

`f(E,E_f)` is the Fermi-Dirac distribution function. The order of the Chebyshev expansion
is `order`. `T` is the temperature
and `Ef` the Fermi energy.

 """
thermalaverageKPM(h::AbstractMatrix, A::AbstractMatrix; ket = missing, kw...) = thermalaverageKPM(momentaKPM(h, obs = A, ket = ket; kw...); kw...)

function thermalaverageKPM(momenta::MomentaKPM{T}; temp = 0.0, Ef = 0.0, kw...) where {T} 
    (center, halfwidth) = momenta.bandbracket
    Ef = Ef * halfwidth + center
    dim = length(momenta.μlist)
    @warn "Computing the integrals..."
    meanlist = [_intαn(n,Ef,temp) for n in 0:dim-1]
    jackson!(meanlist) 
    return halfwidth * sum(meanlist .* (momenta.μlist .* halfwidth .+ center))    #Rescale A in energy units. Potential problem
end

_intαn(n,Ef,temp) = QuadGK.quadgk(E -> αn(n,E,Ef,temp), -1., 1., atol= 1e-10, rtol=1e-10)[1]

αn(n,E,Ef,temp) = fermifun(E,Ef,temp) * 2/(π*(1-E^2)^(-1/2)) * chebypol(n,E) / (1+(n==0 ? 1 : 0))

fermifun(E,Ef,temp) = temp==0 ? (E<Ef ? 1 : 0) : (1/(1+exp((E-0.)/(8.6173324*10^-2*temp))))   

function chebypol(m,x) 
    cheby0=1.0
    cheby1=x
    if m==0
        chebym = cheby0
    elseif m == 1
        chebym = cheby1
    else
        for i in 2:m
            chebym = 2*x*cheby1 - cheby0
            cheby0, cheby1 = cheby1, chebym
        end
    end
    return chebym
end
 
#  function thermalmean(momenta::MomentaKPM{T}; resolution = 2)
#     (center, halfwidth) = momenta.bandbracket
