#######################################################################
# Onsite
#######################################################################
const zerovec = SVector(0.,0.,0.,0.)

abstract type Onsite{SL,N} <: ModelTerm end  #N is the number of orbitals

struct NoOnsite <: Onsite{Missing,0}
end
(o::NoOnsite)(r::SVector{E,T}, ::Val{N}) where {E,T,N} = zero(SMatrix{N,N,T})

struct OnsiteFunc{SL,N,F<:Function} <: Onsite{SL,N}
    f::F
    s::SL  # Sublattices
end
OnsiteFunc{SL,N}(f::F, s::SL) where {N,SL,F} = OnsiteFunc{SL,N,F}(f, s)
(o::OnsiteFunc)(r, ::Val{N}) where {N} = o(r)
(o::OnsiteFunc)(r::SVector) = o.f(r)

struct OnsiteConst{SL,N,T,NN} <: Onsite{SL,N}
    o::SMatrix{N,N,T,NN}
    s::SL  # Sublattices
end
(o::OnsiteConst)(r, ::Val{N}) where {N} = o(r)
(o::OnsiteConst)(r::SVector) = o.o

Onsite(o, s::Vararg{Int, N}) where {N} = _Onsite(o, to_ints_or_missing(s))
_Onsite(f::Function, s) = _OnsiteFunc(f, f(zerovec), s)
    _OnsiteFunc(f::Function, sample::SMatrix{N,N}, s) where {N} = _OnsiteFunc(Val(N), f, s)
    _OnsiteFunc(f::Function, sample, s) = _OnsiteFunc(Val(1), r -> @SMatrix[f(r)], s)
    _OnsiteFunc(::Val{N}, f::F, s::SL) where {SL,N,F<:Function} = OnsiteFunc{SL,N,F}(f, s)
_Onsite(v::SMatrix, s) = OnsiteConst(v, s)
_Onsite(v::T, s) where {T<:Number} = OnsiteConst(SMatrix{1,1,T}(v), s)
function _Onsite(v::AbstractMatrix{T}, s) where T<:Number
    n,m = size(v)
    return OnsiteConst(SMatrix{n,n,T,n*n}(v), s)
end
function _Onsite(v::AbstractVector{T}, s) where T<:Number
    return OnsiteConst(SMatrix{1,1,T,1}(v), s)
end

onsitedims(::Onsite{SL,N}) where {SL,N} = N

#######################################################################
# Hopping
#######################################################################

abstract type Hopping{SL,N,M} <: ModelTerm end

struct NoHopping <: Hopping{Missing,0,0}
end
(h::NoHopping)(rdr::Tuple{S,S}, ::Val{M}, ::Val{N}) where {E,T,M,N,S<:SVector{E,T}} = zero(SMatrix{M,N,T})

struct HoppingFunc{SL,N,M,F<:Function} <: Hopping{SL,N,M}
    f::F
    ss::SL   # Sublattices
end
HoppingFunc{SL,N,M}(f::F, ss::SL) where {N,M,SL,F} = HoppingFunc{SL,N,M,F}(f, ss)
(h::HoppingFunc)(rdr, ::Val{M}, ::Val{N}) where {M,N} = h(rdr)
(h::HoppingFunc)((r, dr)::Tuple{S,S}) where {S<:SVector} = h.f(r, dr)

struct HoppingConst{SL,N,M,T,NM} <: Hopping{SL,N,M}
    h::SMatrix{N,M,T,NM}
    ss::SL   # Sublattices
end
(h::HoppingConst)(rdr, ::Val{M}, ::Val{N}) where {M,N} = h(rdr)
(h::HoppingConst)((r, dr)::Tuple{S,S}) where {S<:SVector} = h.h

Hopping(h, ss::Vararg{Union{Int,Tuple{Int,Int}}, N}) where {N} = _Hopping(h, to_tuples_or_missing(ss))
_Hopping(f::Function, ss) = _HoppingFunc(f, f(zerovec, zerovec), ss)
    _HoppingFunc(f::Function, sample::SMatrix{N,M}, ss) where {N,M} = _HoppingFunc(Val(N), Val(M), f, ss)
    _HoppingFunc(f::Function, sample, ss) = _HoppingFunc(Val(1), Val(1), (r, dr) -> @SMatrix[f(r, dr)], ss)
    _HoppingFunc(::Val{N}, ::Val{M}, f::F, ss::SL) where {SL,N,M,F<:Function} = HoppingFunc{SL,N,M,F}(f, ss)
_Hopping(v::T, ss) where {T<:Number} = HoppingConst(SMatrix{1,1,T}(v), ss)
_Hopping(v::SMatrix{N,M}, ss) where {N,M} = HoppingConst(v, ss)
function _Hopping(v::AbstractMatrix{T}, ss) where {T<:Number}
    n,m = size(v)
    return HoppingConst(SMatrix{n,m,T,n*m}(v), ss)
end
function _Hopping(v::AbstractVector{T}, ss) where {T<:Number,S}
    n = length(v)
    return HoppingConst(SMatrix{n,1,T,n}(v), ss)
end

hoppingdims(::Hopping{SL,N,M}) where {SL,N,M} = (N,M)

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
_model(os::Tuple, hs::Tuple, defo::Tuple, defh::Tuple, o::Onsite{<:Tuple}, terms...) = 
    _model(tuplejoin(os, (o,)), hs, defo, defh, terms...)
_model(os::Tuple, hs::Tuple, defo::Tuple, defh::Tuple, h::Hopping{<:Tuple}, terms...) = 
    _model(os, tuplejoin(hs, (h,)), defo, defh, terms...)
_model(os::Tuple, hs::Tuple, defo::Tuple, defh::Tuple, o::Onsite{Missing}, terms...) = 
    _model(os, hs, (o,), defh, terms...)
_model(os::Tuple, hs::Tuple, defo::Tuple, defh::Tuple, h::Hopping{Missing}, terms...) = 
    _model(os, hs, defo, (h,), terms...)

function _model(os::OS, hs::HS, (defons,)::Tuple{O}, (defhop,)::Tuple{H}) where {OS<:Tuple, HS<:Tuple, O<:Onsite{Missing},H<:Hopping{Missing}}
    defdim = onsitedims(defons)
    subs = 0
    for o in os
        subs = max(subs, maximum(o.s))
    end
    for h in hs
        subs = max(subs, tuplemaximum(h.ss))
    end
    hptr = zeros(Int, subs, subs)
    optr = zeros(Int, subs)
    dims = zeros(Int, subs)
    
    for (i, o) in enumerate(os)
        odims = onsitedims(o)
        for s in o.s
            optr[s] = i
            dims[s] = odims
        end
    end
    for (i, h) in enumerate(hs)
        for ss in h.ss
            hptr[ss[1], ss[2]] = i
            checkdims!(dims, h) 
        end    
    end
    
    defdim = checkdefaultdim(defdim, defhop, dims)
    for (i,dim) in enumerate(dims)
        dim == 0 && (dims[i] = defdim)
    end
    Model{OS,HS,O,H}(os, hs, optr, hptr, dims, defons, defhop, defdim)
end

function checkdims!(dims, h::Hopping{<:Tuple})
    for (s2, s1) in h.ss
        d2, d1 = dims[s2], dims[s1]
        hd2, hd1 = hoppingdims(h)
        s2 != s1 || hd2 == hd1 || throw(DimensionMismatch("same-sublattice hopping must be a square matrix or scalar"))
        d2 == hd2 || (d2 == 0 ? (dims[s2] = hd2) : throw(DimensionMismatch("inconsistent model dimensions")))
        d1 == hd1 || (d1 == 0 ? (dims[s1] = hd1) : throw(DimensionMismatch("inconsistent model dimensions")))
    end
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

nsublats(m::Model) = length(m.dims)

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
    (i > ns ? m.defdim : m.dims[i] for i in 1:nl)
end

dagger(h::NoHopping) = h
dagger(h::HoppingFunc{SL,N,M}) where {SL,N,M} = HoppingFunc{SL,N,M}(daggerF2(h.f), h.ss)
dagger(h::HoppingConst) = HoppingConst(h.h', h.ss)
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

nzonsites(m::Model) = vectornonzeros(m.optr)
nzhoppings(m::Model) = matrixnonzeros(m.hptr)
hasdefaultonsite(m::Model) = isa(m.defonsite, NoOnsite) ? "No" : "Yes"
hasdefaulthopping(m::Model) = isa(m.defhopping, NoHopping) ? "No" : "Yes"