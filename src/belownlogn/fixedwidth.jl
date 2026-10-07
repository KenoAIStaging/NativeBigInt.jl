# Exact Gaussian-dyadic arithmetic for the sub-n-log-n multiplier.
#
# Every numerical value in the construction is a Gaussian dyadic
# 2^-p (a + b i) with integer numerators a, b (paper, Section 2: "fixed-width
# signed fixed-point binary addition, subtraction, sign change, and
# multiplication or division by a fixed power of two are streaming operations
# of linear bit cost, provided enough integer and fractional guard bits have
# been allocated for exact results").  FW{L} is that fixed-width signed word:
# L limbs of two's complement, little-endian, as an immutable tuple so that
# arrays of coefficients are flat and the arithmetic allocates nothing.  The
# width L is chosen per call from p and the guard analysis of the caller.

struct FW{L}
    limbs::NTuple{L,UInt64}
end

const ALLONES = ~UInt64(0)

@inline fw_limbs(::Type{FW{L}}) where {L} = L
@inline fw_limbs(::FW{L}) where {L} = L
@inline nbits(::Type{FW{L}}) where {L} = 64L

@inline Base.zero(::Type{FW{L}}) where {L} = FW{L}(ntuple(_ -> UInt64(0), Val(L)))
@inline Base.zero(::FW{L}) where {L} = zero(FW{L})
@inline Base.iszero(x::FW{L}) where {L} = all(iszero, x.limbs)
@inline Base.signbit(x::FW{L}) where {L} = (@inbounds x.limbs[L]) >> 63 == 1
@inline Base.:(==)(a::FW{L}, b::FW{L}) where {L} = a.limbs == b.limbs

# Sign-extend a machine integer.
@inline function FW{L}(x::Int64) where {L}
    ext = x < 0 ? ALLONES : UInt64(0)
    FW{L}(ntuple(i -> i == 1 ? reinterpret(UInt64, x) : ext, Val(L)))
end
@inline function FW{L}(x::Int128) where {L}
    ext = x < 0 ? ALLONES : UInt64(0)
    u = reinterpret(UInt128, x)
    FW{L}(ntuple(i -> i == 1 ? UInt64(u & typemax(UInt64)) : i == 2 ? UInt64(u >> 64) : ext, Val(L)))
end

# Change width: sign-extend when growing, drop high limbs when shrinking
# (the caller guarantees the value fits).
@inline function resize(::Type{FW{L2}}, x::FW{L}) where {L2,L}
    ext = signbit(x) ? ALLONES : UInt64(0)
    FW{L2}(ntuple(i -> i <= L ? (@inbounds x.limbs[i]) : ext, Val(L2)))
end

# Straight-line carry chains and schoolbook products are generated per width
# (a closure that mutates its carry would be boxed by the compiler).
@generated function Base.:+(a::FW{L}, b::FW{L}) where {L}
    ex = Expr(:block, :(la = a.limbs), :(lb = b.limbs), :(c = UInt64(0)))
    for i in 1:L
        s = Symbol(:s, i)
        push!(ex.args, quote
            ($s, o1) = Base.add_with_overflow(la[$i], lb[$i])
            ($s, o2) = Base.add_with_overflow($s, c)
            c = UInt64(o1) | UInt64(o2)
        end)
    end
    push!(ex.args, :(FW{$L}(($([Symbol(:s, i) for i in 1:L]...),))))
    return ex
end

@generated function Base.:-(a::FW{L}, b::FW{L}) where {L}
    ex = Expr(:block, :(la = a.limbs), :(lb = b.limbs), :(brw = UInt64(0)))
    for i in 1:L
        s = Symbol(:s, i)
        push!(ex.args, quote
            ($s, o1) = Base.sub_with_overflow(la[$i], lb[$i])
            ($s, o2) = Base.sub_with_overflow($s, brw)
            brw = UInt64(o1) | UInt64(o2)
        end)
    end
    push!(ex.args, :(FW{$L}(($([Symbol(:s, i) for i in 1:L]...),))))
    return ex
end

@inline Base.:-(a::FW{L}) where {L} = zero(FW{L}) - a
@inline Base.:~(a::FW{L}) where {L} = FW{L}(map(~, a.limbs))

@inline function Base.:<<(x::FW{L}, k::Int) where {L}
    k <= 0 && return x
    k >= 64L && return zero(FW{L})
    q, r = divrem(k, 64)
    t = ntuple(Val(L)) do i
        j = i - q
        lo = j >= 1 ? (@inbounds x.limbs[j]) : UInt64(0)
        r == 0 && return lo
        hi = j >= 2 ? (@inbounds x.limbs[j-1]) : UInt64(0)
        (lo << r) | (hi >> (64 - r))
    end
    FW{L}(t)
end

# Arithmetic (sign-propagating) right shift.
@inline function Base.:>>(x::FW{L}, k::Int) where {L}
    k <= 0 && return x
    ext = signbit(x) ? ALLONES : UInt64(0)
    k >= 64L && return FW{L}(ntuple(_ -> ext, Val(L)))
    q, r = divrem(k, 64)
    t = ntuple(Val(L)) do i
        j = i + q
        lo = j <= L ? (@inbounds x.limbs[j]) : ext
        r == 0 && return lo
        hi = j + 1 <= L ? (@inbounds x.limbs[j+1]) : ext
        (lo >> r) | (hi << (64 - r))
    end
    FW{L}(t)
end

# Truncation toward zero of x / 2^k (the paper's Q_p when k is the number
# of surplus fractional bits).
@inline trunc_shr(x::FW{L}, k::Int) where {L} = signbit(x) ? -((-x) >> k) : x >> k

# Nearest integer to x / 2^k.  Used only where the error is strictly below
# 1/2, so ties cannot occur; floor(x/2^k + 1/2) is then exact.
@inline function round_shr(x::FW{L}, k::Int) where {L}
    k == 0 && return x
    (x + (FW{L}(1) << (k - 1))) >> k
end

# Wrapping schoolbook product modulo 2^(64L).  For two's complement inputs
# the low 64L bits of the signed product are exactly the low bits of the
# unsigned product of the representatives, so a caller that widens both
# operands to a width holding the exact product gets it exactly.
@generated function Base.:*(a::FW{L}, b::FW{L}) where {L}
    ex = Expr(:block, :(la = a.limbs), :(lb = b.limbs))
    for k in 1:L
        push!(ex.args, :($(Symbol(:acc, k)) = UInt64(0)))
    end
    for i in 1:L
        push!(ex.args, :(ai = la[$i]), :(c = UInt64(0)))
        for j in 1:L-i+1
            acc = Symbol(:acc, i + j - 1)
            push!(ex.args, quote
                pr = widemul(ai, lb[$j]) + $acc + c
                $acc = pr % UInt64
                c = (pr >> 64) % UInt64
            end)
        end
    end
    push!(ex.args, :(FW{$L}(($([Symbol(:acc, k) for k in 1:L]...),))))
    return ex
end

# Exact signed product of two L-limb words in L2 >= 2L limbs.
@inline mul_wide(::Type{FW{L2}}, a::FW{L}, b::FW{L}) where {L2,L} =
    resize(FW{L2}, a) * resize(FW{L2}, b)

# Multiply by a machine integer, wrapping at 64L bits (callers keep it in
# range).
@inline Base.:*(a::FW{L}, c::Int64) where {L} = a * FW{L}(c)
@inline Base.:*(c::Int64, a::FW{L}) where {L} = a * FW{L}(c)

function Base.cmp(a::FW{L}, b::FW{L}) where {L}
    sa, sb = signbit(a), signbit(b)
    sa != sb && return sa ? -1 : 1
    @inbounds for i in L:-1:1
        a.limbs[i] != b.limbs[i] && return a.limbs[i] < b.limbs[i] ? -1 : 1
    end
    return 0
end
Base.isless(a::FW{L}, b::FW{L}) where {L} = cmp(a, b) < 0
Base.:<(a::FW{L}, b::FW{L}) where {L} = cmp(a, b) < 0
Base.:<=(a::FW{L}, b::FW{L}) where {L} = cmp(a, b) <= 0
Base.abs(a::FW{L}) where {L} = signbit(a) ? -a : a

# Bit length of |x| (0 for x == 0); the caller checks fits(x, bits) before
# narrowing.
function magbits(x::FW{L}) where {L}
    m = abs(x)
    @inbounds for i in L:-1:1
        m.limbs[i] != 0 && return 64 * (i - 1) + Base.top_set_bit(m.limbs[i])
    end
    return 0
end
# x representable as a signed `bits`-bit two's complement word
fits(x::FW{L}, bits::Int) where {L} = magbits(x) < bits || (signbit(x) && magbits(x) == bits && iszero((-x) & ((FW{L}(1) << (bits - 1)) - FW{L}(1))))

@inline Base.:&(a::FW{L}, b::FW{L}) where {L} = FW{L}(map(&, a.limbs, b.limbs))

# Conversions used by tests, parameter setup and the limb-array boundaries.
function Base.BigInt(x::FW{L}) where {L}
    neg = signbit(x)
    m = neg ? -x : x
    r = big(0)
    @inbounds for i in L:-1:1
        r = (r << 64) + m.limbs[i]
    end
    return neg ? -r : r
end
function FW{L}(x::BigInt) where {L}
    neg = x < 0
    m = abs(x)
    t = ntuple(Val(L)) do i
        UInt64((m >> (64 * (i - 1))) & typemax(UInt64))
    end
    v = FW{L}(t)
    return neg ? -v : v
end
Base.Int64(x::FW{L}) where {L} = reinterpret(Int64, @inbounds x.limbs[1])
function Base.Int128(x::FW{L}) where {L}
    L == 1 && return Int128(Int64(x))
    reinterpret(Int128, (UInt128(@inbounds x.limbs[2]) << 64) | UInt128(@inbounds x.limbs[1]))
end

# Load the two's complement word stored at mem[off+1 .. off+n] (n limbs,
# little-endian, sign in the top limb) into an FW{L}.
@inline function load_fw(::Type{FW{L}}, mem::Memory{UInt64}, off::Int, n::Int) where {L}
    ext = (@inbounds mem[off+n]) >> 63 == 1 ? ALLONES : UInt64(0)
    FW{L}(ntuple(i -> i <= n ? (@inbounds mem[off+i]) : ext, Val(L)))
end
# Store the low n limbs (sign-extended as needed) of x at mem[off+1 .. off+n].
@inline function store_fw!(mem::Memory{UInt64}, off::Int, n::Int, x::FW{L}) where {L}
    ext = signbit(x) ? ALLONES : UInt64(0)
    @inbounds for i in 1:n
        mem[off+i] = i <= L ? x.limbs[i] : ext
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Gaussian dyadics 2^-p (re + im i) with FW{L} numerators.

struct GC{L}
    re::FW{L}
    im::FW{L}
end

@inline Base.zero(::Type{GC{L}}) where {L} = GC{L}(zero(FW{L}), zero(FW{L}))
@inline Base.iszero(z::GC{L}) where {L} = iszero(z.re) && iszero(z.im)
@inline Base.:+(a::GC{L}, b::GC{L}) where {L} = GC{L}(a.re + b.re, a.im + b.im)
@inline Base.:-(a::GC{L}, b::GC{L}) where {L} = GC{L}(a.re - b.re, a.im - b.im)
@inline Base.:-(a::GC{L}) where {L} = GC{L}(-a.re, -a.im)
@inline Base.conj(a::GC{L}) where {L} = GC{L}(a.re, -a.im)
@inline Base.:(==)(a::GC{L}, b::GC{L}) where {L} = a.re == b.re && a.im == b.im
# multiplication by i swaps the components and changes one sign
@inline mul_i(a::GC{L}) where {L} = GC{L}(-a.im, a.re)
@inline mul_negi(a::GC{L}) where {L} = GC{L}(a.im, -a.re)
# i^k for k mod 4
@inline function mul_ipow(a::GC{L}, k::Int) where {L}
    k &= 3
    k == 0 ? a : k == 1 ? mul_i(a) : k == 2 ? -a : mul_negi(a)
end
@inline Base.:<<(a::GC{L}, k::Int) where {L} = GC{L}(a.re << k, a.im << k)
@inline Base.:>>(a::GC{L}, k::Int) where {L} = GC{L}(a.re >> k, a.im >> k)
@inline trunc_shr(a::GC{L}, k::Int) where {L} = GC{L}(trunc_shr(a.re, k), trunc_shr(a.im, k))
@inline resize(::Type{GC{L2}}, a::GC{L}) where {L2,L} = GC{L2}(resize(FW{L2}, a.re), resize(FW{L2}, a.im))
@inline Base.:*(a::GC{L}, c::Int64) where {L} = GC{L}(a.re * c, a.im * c)

# Exact product in a wider width (L2 >= 2L).
@inline function mul_wide(::Type{GC{L2}}, a::GC{L}, b::GC{L}) where {L2,L}
    ar, ai = resize(FW{L2}, a.re), resize(FW{L2}, a.im)
    br, bi = resize(FW{L2}, b.re), resize(FW{L2}, b.im)
    GC{L2}(ar * br - ai * bi, ar * bi + ai * br)
end

# Q_p of a product of two p-bit grid values: exact product, then truncate
# the surplus p fractional bits toward zero and narrow back.
@inline function mul_trunc(a::GC{L}, b::GC{L}, p::Int, ::Type{GC{L2}}) where {L,L2}
    w = mul_wide(GC{L2}, a, b)
    GC{L}(resize(FW{L}, trunc_shr(w.re, p)), resize(FW{L}, trunc_shr(w.im, p)))
end
# Same with a real factor
@inline function mul_trunc(a::GC{L}, c::FW{L}, p::Int, ::Type{GC{L2}}) where {L,L2}
    cw = resize(FW{L2}, c)
    GC{L}(resize(FW{L}, trunc_shr(resize(FW{L2}, a.re) * cw, p)),
          resize(FW{L}, trunc_shr(resize(FW{L2}, a.im) * cw, p)))
end

# Widening type for exact products of two L-limb words.
wide_type(::Type{GC{L}}) where {L} = GC{2L + 1}
wide_type(::Type{FW{L}}) where {L} = FW{2L + 1}

# Test helpers: exact complex value as a pair of BigInt numerators.
tobig(z::GC{L}) where {L} = (BigInt(z.re), BigInt(z.im))
GC{L}(re::BigInt, im::BigInt) where {L} = GC{L}(FW{L}(re), FW{L}(im))
