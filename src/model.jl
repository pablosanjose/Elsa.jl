#######################################################################
# Onsite
#######################################################################
const zerovec = SVector(0.,0.,0.,0.)

abstract type Onsite{IM<:Union{Int,Missing},N} <: ModelTerm end  #N is the number of orbitals

struct NoOnsite <: Onsite{Missing,0}
end
@inline (o::NoOnsite)(r::SVector{E,T}, ::Val{N}) where {E,T,N} = zero(SMatrix{N,N,T})

struct OnsiteFunc{IM,N,F<:Function} <: Onsite{IM,N}
    s::IM
    f::F
end
OnsiteFunc{IM,N}(s::IM, f::F) where {N,IM,F} = OnsiteFunc{IM,N,F}(s, f)
@inline (o::OnsiteFunc)(r, ::Val{N}) where {N} = o(r)
(o::OnsiteFunc)(r::SVector) = o.f(r)

struct OnsiteConst{IM,N,T,NN} <: Onsite{IM,N}
    s::IM  # Sublattice
    o::SMatrix{N,N,T,NN}
end
@inline (o::OnsiteConst)(r, ::Val{N}) where {N} = o(r)
(o::OnsiteConst)(r::SVector) = o.o

Onsite(arg::Union{Function, AbstractArray, Number}) = Onsite(missing, arg)  # Default
Onsite(s, f::Function) = _OnsiteFunc(s, f, f(zerovec))
_OnsiteFunc(s, f::Function, sample::SMatrix{N,N}) where {N} = _OnsiteFunc(Val(N), s, f)
_OnsiteFunc(s, f::Function, sample) = _OnsiteFunc(Val(1), s, r -> @SMatrix[f(r)])
_OnsiteFunc(::Val{N}, s::IM, f::F) where {IM,N,F<:Function} = OnsiteFunc{IM,N,F}(s,f)
Onsite(s, v::SMatrix) = OnsiteConst(s,v)
Onsite(s, v::T) where {T<:Number} = OnsiteConst(s, SMatrix{1,1,T}(v))
function Onsite(s, v::AbstractMatrix{T}) where T<:Number
    n,m = size(v)
    return OnsiteConst(s, SMatrix{n,n,T,n*n}(v))
end
#To allow Onsite(1,[2])
function Onsite(s, v::AbstractVector{T}) where T<:Number
    return OnsiteConst(s, SMatrix{1,1,T,1}(v))
end


onsitedims(::Onsite{IM,N}) where {IM,N} = N

#######################################################################
# Hopping
#######################################################################

abstract type Hopping{IM<:Union{Tuple{Int,Int},Missing},N,M} <: ModelTerm end

struct NoHopping <: Hopping{Missing,0,0}
end
@inline (h::NoHopping)(rdr::Tuple{S,S}, ::Val{M}, ::Val{N}) where {E,T,M,N,S<:SVector{E,T}} = zero(SMatrix{M,N,T})

struct HoppingFunc{IM,N,M,F<:Function} <: Hopping{IM,N,M}
    ss::IM   # Sublattices
    f::F
end
HoppingFunc{IM,N,M}(ss::IM, f::F) where {N,M,IM,F} = HoppingFunc{IM,N,M,F}(ss, f)
@inline (h::HoppingFunc)(rdr, ::Val{M}, ::Val{N}) where {M,N} = h(rdr)
(h::HoppingFunc)((r, dr)::Tuple{S,S}) where {S<:SVector} = h.f(r, dr)

struct HoppingConst{IM,N,M,T,NM} <: Hopping{IM,N,M}
    ss::IM   # Sublattices
    h::SMatrix{N,M,T,NM}
end
@inline (h::HoppingConst)(rdr, ::Val{M}, ::Val{N}) where {M,N} = h(rdr)
(h::HoppingConst)((r, dr)::Tuple{S,S}) where {S<:SVector} = h.h

Hopping(arg::Union{Function, AbstractArray, Number}) = Hopping(missing, arg)  # Default
Hopping(ss, f::Function) = _HoppingFunc(ss, f, f(zerovec, zerovec))
_HoppingFunc(ss, f::Function, sample::SMatrix{N,M}) where {N,M} = _HoppingFunc(Val(N), Val(M), ss, f)
_HoppingFunc(ss, f::Function, sample) = _HoppingFunc(Val(1), Val(1), ss, (r, dr) -> @SMatrix[f(r, dr)])
_HoppingFunc(::Val{N}, ::Val{M}, ss::IM, f::F) where {IM,N,M,F<:Function} = HoppingFunc{IM,N,M,F}(ss,f)
Hopping(ss, v::T) where {T<:Number} = HoppingConst(ss, SMatrix{1,1,T}(v))
Hopping(ss, v::SMatrix{N,M}) where {N,M} = HoppingConst(ss, v)
function Hopping(ss, v::AbstractMatrix{T}) where T<:Number
    n,m = size(v)
    return HoppingConst(ss, SMatrix{n,m,T,n*m}(v))
end
function Hopping(ss, v::AbstractVector{T}) where T<:Number
    n = length(v)
    return HoppingConst(ss, SMatrix{n,1,T,n}(v))
end

hoppingdims(::Hopping{IM,N,M}) where {IM,N,M} = (N,M)

#######################################################################
# Model
#######################################################################

struct Model{OS<:Tuple, HS<:Tuple, O, H}
    onsites::OS
    hoppings::HS
    optr::Vector{Int}
    hptr::Matrix{Int}    
    dims::Vector{Int}
    defonsite::O
    defhopping::H
    defdim::Int
end

Model(terms...) = _model((), (), (NoOnsite(),), (NoHopping(),), terms...)
_model(os::Tuple, hs::Tuple, defo::Tuple, defh::Tuple, o::Onsite{Int}, terms...) = 
    _model(tuplejoin(os, (o,)), hs, defo, defh, terms...)
_model(os::Tuple, hs::Tuple, defo::Tuple, defh::Tuple, h::Hopping{Tuple{Int,Int}}, terms...) = 
    _model(os, tuplejoin(hs, (h,)), defo, defh, terms...)
_model(os::Tuple, hs::Tuple, defo::Tuple, defh::Tuple, o::Onsite{Missing}, terms...) = 
    _model(os, hs, (o,), defh, terms...)
_model(os::Tuple, hs::Tuple, defo::Tuple, defh::Tuple, h::Hopping{Missing}, terms...) = 
    _model(os, hs, defo, (h,), terms...)

function _model(os::OS, hs::HS, (defons,)::Tuple{O}, (defhop,)::Tuple{H}) where {OS<:Tuple, HS<:Tuple, O<:Onsite{Missing},H<:Hopping{Missing}}
    defdim = onsitedims(defons)
    subs = 0
    for o in os
        subs = max(subs, o.s)
    end
    for h in hs
        subs = max(subs, h.ss[1], h.ss[2])
    end
    hptr = zeros(Int, subs, subs)
    optr = zeros(Int, subs)
    dims = zeros(Int, subs)
    
    for (i, o) in enumerate(os)
        optr[o.s] = i
        dims[o.s] = onsitedims(o)
    end
    for (i, h) in enumerate(hs)
        hptr[h.ss[1], h.ss[2]] = i
        checkdims!(dims, h)     
    end
    
    defdim = checkdefaultdim(defdim, defhop, dims)
    for (i,dim) in enumerate(dims)
        dim == 0 && (dims[i] = defdim)
    end
    Model{OS,HS,O,H}(os, hs, optr, hptr, dims, defons, defhop, defdim)
end

function checkdims!(dims, h::Hopping{Tuple{Int,Int}})
    s2, s1 = h.ss
    d2, d1 = dims[s2], dims[s1]
    hd2, hd1 = hoppingdims(h)
    s2 != s1 || hd2 == hd1 || throw(DimensionMismatch("same-sublattice hopping must be a square matrix or scalar"))
    d2 == hd2 || (d2 == 0 ? (dims[s2] = hd2) : throw(DimensionMismatch("inconsistent model dimensions")))
    d1 == hd1 || (d1 == 0 ? (dims[s1] = hd1) : throw(DimensionMismatch("inconsistent model dimensions")))
    return
end
checkdefaultdim(defdim, defh::NoHopping, dims) = defdim
function checkdefaultdim(defdim, defh::Hopping, dims)
    hd1, hd2 = hoppingdims(defh)
    hd1 == hd2 || throw(DimensionMismatch("default hopping must be a finite-sized square matrix or scalar"))
    defdim == 0 && return hd1
    defdim == hd1 || throw(DimensionMismatch("default hopping has inconsistent dimensions with default onsite"))
    all(dim == 0 || dim == hd1 for dim in dims) || throw(DimensionMismatch("default hopping has inconsistent dimensions with non-default hoppings/onsites"))
    return defdim
end

@inline nsublats(m::Model) = length(m.dims)

function hopping(m::Model, (s2, s1))
    nh = size(m.hptr, 1)
    if 0 < s1 <= nh && 0 < s2 <= nh
        ptr = m.hptr[s2, s1]
        ptr2 = m.hptr[s1, s2]
        if ptr == 0 && ptr2 != 0
            return dagger(m.hoppings[ptr2])
        elseif ptr != 0 
            return m.hoppings[ptr]
        end
    end
    return defaulthopping(m, s2, s1)
end
defaulthopping(m, s2, s1) = 
    if s1 <= s2
        return m.defhopping
    else
        return dagger(m.defhopping)
    end

function onsite(m::Model, s)
    no = length(m.optr)
    if 0 < s <= no
        ptr = m.optr[s]
        if ptr != 0
            return m.onsites[ptr]
        end
    end
    return defaultonsite(m, s)
end
defaultonsite(m, s) = m.defonsite


sublatdims(m::Model) = m.dims
function sublatdims(lat::Lattice, m::Model)
    ns = nsublats(m)
    nl = nsublats(lat)
    # ns <= nl || throw(DimensionMismatch("model contains more non-zero sublats than lattice"))
    (i > ns ? m.defdim : m.dims[i] for i in 1:nl)
end

dagger(h::NoHopping) = h
dagger(h::HoppingFunc{IM,N,M}) where {IM,N,M} = HoppingFunc{IM,N,M}(h.ss, daggerF2(h.f))
dagger(h::HoppingConst) = HoppingConst(h.ss, h.h')
daggerF2(f::Function) = (r,dr) -> f(r,-dr)'

#######################################################################
# Model display
#######################################################################

function Base.show(io::IO, model::Model)
    print(io, "Model with sublattice site dimensions $((model.dims..., )) (default $(model.defdim))
    Sublattice-specific onsites  : $((nzonsites(model)... ,))
    Sublattice-specific hoppings : $((nzhoppings(model)... ,))
    Default onsites  : $(hasdefaultonsite(model))
    Default hoppings : $(hasdefaulthopping(model))")
end

nzonsites(m::Model) = [o.s for o in m.onsites]
nzhoppings(m::Model) = [h.ss for h in m.hoppings]
hasdefaultonsite(m::Model) = isa(m.defonsite, NoOnsite) ? "No" : "Yes"
hasdefaulthopping(m::Model) = isa(m.defhopping, NoHopping) ? "No" : "Yes"