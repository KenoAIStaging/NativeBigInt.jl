# NBig type definition and interface (mpz layer)

struct NBig <: Signed
    signlen::Int           # sign(x) * limb count; 0 ⟺ x == 0
    limbs::Memory{Limb}    # little-endian, normalized (top limb ≠ 0), may be over-allocated
end

const EMPTY_LIMBS = Memory{Limb}(undef, 0)

@inline nlimbs(x::NBig) = abs(x.signlen)
@inline Base.sign(x::NBig) = sign(x.signlen)
@inline Base.signbit(x::NBig) = x.signlen < 0
@inline Base.iszero(x::NBig) = x.signlen == 0

function nbig_from_limbs(sgn::Int, limbs::Memory{Limb}, n::Int)
    n = normlen(limbs, 0, n)
    n == 0 && return NBig(0, EMPTY_LIMBS)
    return NBig(sgn * n, limbs)
end

function NBig(x::Base.BitInteger)
    neg = signbit(x)
    mag = neg ? -unsigned(x) : unsigned(x)
    mag == 0 && return NBig(0, EMPTY_LIMBS)
    n = cld(Base.top_set_bit(mag), 64)
    limbs = Memory{Limb}(undef, n)
    @inbounds for i in 1:n
        limbs[i] = (mag >> (64 * (i - 1))) % Limb
    end
    return NBig(neg ? -n : n, limbs)
end

NBig(x::Integer) = NBig(BigInt(x))
NBig(x::NBig) = x

# mpz_import/mpz_export with explicit 8-byte native-endian words are
# independent of GMP's limb size (32-bit on Windows), so both directions
# are a linear copy.
function NBig(x::BigInt)
    x.size == 0 && return NBig(0, EMPTY_LIMBS)
    nbits = ccall((:__gmpz_sizeinbase, :libgmp), Csize_t, (Ref{BigInt}, Cint), x, 2)
    n = (Int(nbits) + 63) >> 6
    limbs = Memory{Limb}(undef, n)
    ccall((:__gmpz_export, :libgmp), Ptr{Limb},
          (Ptr{Limb}, Ptr{Csize_t}, Cint, Csize_t, Cint, Csize_t, Ref{BigInt}),
          limbs, C_NULL, -1, sizeof(Limb), 0, 0, x)
    return NBig(sign(Int(x.size)) * n, limbs)
end

function Base.BigInt(x::NBig)
    n = nlimbs(x)
    n == 0 && return big(0)
    r = BigInt()
    ccall((:__gmpz_import, :libgmp), Cvoid,
          (Ref{BigInt}, Csize_t, Cint, Csize_t, Cint, Csize_t, Ptr{Limb}),
          r, n, -1, sizeof(Limb), 0, 0, x.limbs)
    return signbit(x) ? Base.GMP.MPZ.neg!(r) : r
end

function Base.:(==)(a::NBig, b::NBig)
    a.signlen != b.signlen && return false
    n = nlimbs(a)
    for i in 1:n
        a.limbs[i] != b.limbs[i] && return false
    end
    return true
end

function Base.cmp(a::NBig, b::NBig)
    sa, sb = sign(a.signlen), sign(b.signlen)
    sa != sb && return sa < sb ? -1 : 1
    c = cmp_limbs(a.limbs, 0, nlimbs(a), b.limbs, 0, nlimbs(b))
    return sa < 0 ? -c : c
end
Base.isless(a::NBig, b::NBig) = cmp(a, b) < 0
Base.:<(a::NBig, b::NBig) = cmp(a, b) < 0

Base.:-(x::NBig) = iszero(x) ? x : NBig(-x.signlen, x.limbs)
Base.abs(x::NBig) = iszero(x) || !signbit(x) ? x : NBig(-x.signlen, x.limbs)

Base.isodd(x::NBig) = nlimbs(x) > 0 && isodd(x.limbs[1])
Base.iseven(x::NBig) = !isodd(x)

# Magnitude add/sub; the result sign is decided here from operand signs and
# magnitude comparison, then magnitudes combine via the kernels.
function Base.:+(a::NBig, b::NBig)
    iszero(a) && return b
    iszero(b) && return a
    la, lb = nlimbs(a), nlimbs(b)
    if sign(a) == sign(b)
        if la < lb
            a, b = b, a
            la, lb = lb, la
        end
        r = Memory{Limb}(undef, la + 1)
        c = add!(r, 0, a.limbs, 0, la, b.limbs, 0, lb)
        @inbounds r[la+1] = c
        return nbig_from_limbs(sign(a), r, la + 1)
    else
        c = cmp_limbs(a.limbs, 0, la, b.limbs, 0, lb)
        c == 0 && return NBig(0, EMPTY_LIMBS)
        if c < 0
            a, b = b, a
            la, lb = lb, la
        end
        r = Memory{Limb}(undef, la)
        sub!(r, 0, a.limbs, 0, la, b.limbs, 0, lb)
        return nbig_from_limbs(sign(a), r, la)
    end
end
Base.:-(a::NBig, b::NBig) = a + (-b)

function Base.:*(a::NBig, b::NBig)
    (iszero(a) || iszero(b)) && return NBig(0, EMPTY_LIMBS)
    la, lb = nlimbs(a), nlimbs(b)
    if a.limbs === b.limbs
        # same magnitude (covers x*x and x^2 via power_by_squaring): square it
        r = Memory{Limb}(undef, 2la)
        sqr!(r, 0, a.limbs, 0, la)
        return nbig_from_limbs(sign(a) * sign(b), r, 2la)
    end
    if la < lb
        a, b = b, a
        la, lb = lb, la
    end
    r = Memory{Limb}(undef, la + lb)
    mul!(r, 0, a.limbs, 0, la, b.limbs, 0, lb)
    return nbig_from_limbs(sign(a) * sign(b), r, la + lb)
end

# Truncated division (Base semantics): rem takes the sign of a.
function Base.divrem(a::NBig, b::NBig)
    iszero(b) && throw(DivideError())
    la, lb = nlimbs(a), nlimbs(b)
    cmp_limbs(a.limbs, 0, la, b.limbs, 0, lb) < 0 && return (NBig(0, EMPTY_LIMBS), a)
    q = Memory{Limb}(undef, la - lb + 1)
    r = Memory{Limb}(undef, lb)
    divrem!(q, 0, r, 0, a.limbs, 0, la, b.limbs, 0, lb)
    return (nbig_from_limbs(sign(a) * sign(b), q, la - lb + 1),
            nbig_from_limbs(sign(a), r, lb))
end
Base.div(a::NBig, b::NBig) = divrem(a, b)[1]
Base.rem(a::NBig, b::NBig) = divrem(a, b)[2]

function Base.mod(a::NBig, b::NBig)
    r = rem(a, b)
    return (iszero(r) || sign(r) == sign(b)) ? r : r + b
end
function Base.fld(a::NBig, b::NBig)
    q, r = divrem(a, b)
    return (!iszero(r) && sign(r) != sign(b)) ? q - one(NBig) : q
end
function Base.cld(a::NBig, b::NBig)
    q, r = divrem(a, b)
    return (!iszero(r) && sign(r) == sign(b)) ? q + one(NBig) : q
end

# Base semantics: result = mod(a^n, m), so the sign follows m. The exponent
# is any Integer: powermod_limbs reads it via expbit/expbits, no conversion.
function Base.powermod(a::NBig, n::Integer, m::NBig)
    iszero(m) && throw(DivideError())
    signbit(n) && return powermod(invmod(a, m), -n, m)
    mm = abs(m)
    k = nlimbs(mm)
    k == 1 && (@inbounds mm.limbs[1]) == 1 && return NBig(0, EMPTY_LIMBS)
    b = mod(a, mm)
    r0 = if iszero(n)
        one(NBig)
    elseif iszero(b)
        NBig(0, EMPTY_LIMBS)
    else
        r = powermod_limbs(b.limbs, nlimbs(b), n, mm.limbs, k)
        nbig_from_limbs(1, r, k)
    end
    return signbit(m) && !iszero(r0) ? r0 + m : r0
end

function Base.gcd(a::NBig, b::NBig)
    iszero(a) && return abs(b)
    iszero(b) && return abs(a)
    la, lb = nlimbs(a), nlimbs(b)
    cap = max(la, lb) + 1
    u = Memory{Limb}(undef, cap)
    v = Memory{Limb}(undef, cap)
    copyto!(u, 1, a.limbs, 1, la)
    copyto!(v, 1, b.limbs, 1, lb)
    mem, n = gcd!(u, la, v, lb)
    return nbig_from_limbs(1, mem, n)
end

# Run gcdext! on the magnitudes of (x, y); returns (g, t) as NBigs with
# s*|x| + t*|y| == g for the (recoverable) s.
function gcdext_nbig(x::NBig, y::NBig)
    lx, ly = nlimbs(x), nlimbs(y)
    cap = max(lx, ly) + 1
    u = Memory{Limb}(undef, cap)
    v = Memory{Limb}(undef, cap)
    copyto!(u, 1, x.limbs, 1, lx)
    copyto!(v, 1, y.limbs, 1, ly)
    g, lg, t, lt, tpos = gcdext!(u, lx, v, ly)
    return nbig_from_limbs(1, g, lg), nbig_from_limbs(tpos ? 1 : -1, t, lt)
end

# Extended Lehmer gcd (gcdext! in algorithms.jl) on the magnitudes; the
# a-cofactor is recovered from the identity s = (g - t*|b|) / |a| (exact),
# then both cofactors flip with their operand's sign. Matches Base's
# generic gcdx values (Euclid cofactors are unique given g and t).
function Base.gcdx(a::NBig, b::NBig)
    iszero(b) && return (abs(a), NBig(sign(a)), NBig(0, EMPTY_LIMBS))
    iszero(a) && return (abs(b), NBig(0, EMPTY_LIMBS), NBig(sign(b)))
    g, t = gcdext_nbig(a, b)
    s = div(g - t * abs(b), abs(a))
    return (g, signbit(a) ? -s : s, signbit(b) ? -t : t)
end

# Extended Lehmer gcd tracking only the m-cofactor; |t| < |m|, so one
# conditional add/sub yields Base's sign convention (result follows m).
function Base.invmod(a::NBig, m::NBig)
    iszero(m) && throw(DomainError(m, "`m` must be nonzero."))
    ma = abs(m)
    nlimbs(ma) == 1 && (@inbounds ma.limbs[1]) == 1 && return NBig(0, EMPTY_LIMBS)
    r = mod(a, ma)
    iszero(r) && throw(DomainError((a, m), "Greatest common divisor is $(ma)."))
    g, t = gcdext_nbig(ma, r)
    g == 1 || throw(DomainError((a, m), "Greatest common divisor is $(g)."))
    x = signbit(t) ? t + ma : t
    return signbit(m) && !iszero(x) ? x - ma : x
end

# isqrt: normalize by an even bit shift so sqrtrem!'s precondition
# (top limb >= 2^62) holds, then shift the root back down by half.
function Base.isqrt(x::NBig)
    signbit(x) && throw(DomainError(x, "isqrt requires a nonnegative argument"))
    iszero(x) && return x
    n = nlimbs(x)
    if n <= 2
        hi = n == 2 ? (@inbounds x.limbs[2]) : zero(Limb)
        v = (UInt128(hi) << 64) | (@inbounds x.limbs[1])
        return NBig(isqrt(v) % Limb)
    end
    bits = magnitude_bits(x.limbs, 0, n)
    e = (64n - bits) & ~1
    # sqrtrem! needs an even limb count; odd n gets a zero low limb (a β
    # multiply — a further even shift), undone below via the root shift.
    pad = isodd(n) ? 1 : 0
    nn = n + pad
    h = nn >> 1
    # separate allocations for the operand copy (destroyed by sqrt!) and the
    # scratch: one combined slab crosses Julia's ~2KB allocation-pool limit
    # from 4k bits up (a >2KB Memory costs ~15x a pooled one), exactly where
    # the total runtime is still sub-microsecond.
    buf = Memory{Limb}(undef, nn)
    pad == 1 && (@inbounds buf[1] = 0)
    if e == 0
        copyto!(buf, pad + 1, x.limbs, 1, n)
    else
        lshift!(buf, pad, x.limbs, 0, n, e)
    end
    scratch = Memory{Limb}(undef, sqrt_scratch_len(h))
    s = Memory{Limb}(undef, h)
    sqrt!(s, 0, buf, 0, nn, scratch, 0)
    sh = (e >> 1) + 32pad
    sh > 0 && rshift!(s, 0, s, 0, h, sh)
    return nbig_from_limbs(1, s, h)
end

function Base.:<<(x::NBig, c::UInt)
    (iszero(x) || c == 0) && return x
    lx = nlimbs(x)
    lw, cnt = Int(c >> 6), Int(c & 63)
    n = lx + lw + 1
    r = Memory{Limb}(undef, n)
    fill!(view(r, 1:lw), zero(Limb))
    @inbounds r[n] = lshift!(r, lw, x.limbs, 0, lx, cnt)
    return nbig_from_limbs(sign(x), r, n)
end

function Base.:>>(x::NBig, c::UInt)
    (iszero(x) || c == 0) && return x
    lx = nlimbs(x)
    lw, cnt = Int(c >> 6), Int(c & 63)
    if lw >= lx
        # magnitude fully shifted out: 0, or -1 by flooring
        return signbit(x) ? NBig(-1) : NBig(0, EMPTY_LIMBS)
    end
    n = lx - lw
    r = Memory{Limb}(undef, n)
    lost = rshift!(r, 0, x.limbs, lw, n, cnt) != 0
    if signbit(x) && !lost
        @inbounds for i in 1:lw
            x.limbs[i] != 0 && (lost = true; break)
        end
    end
    q = nbig_from_limbs(sign(x), r, n)
    # arithmetic shift floors: negative with dropped bits rounds away from zero
    return (signbit(x) && lost) ? q - one(NBig) : q
end
Base.:>>>(x::NBig, c::UInt) = x >> c

Base.:~(x::NBig) = -(x + one(NBig))

# In-place two's-complement negation of t[1..n]: t <- ~t + 1.
@inline function negate_twos!(t::Memory{Limb}, n::Int)
    c = Limb(1)
    @inbounds for i in 1:n
        t[i], c = add_limb_c(~t[i], c, zero(Limb))
    end
    return t
end

# Two's-complement image of x in t[1..n]; requires n > nlimbs(x) so the sign
# extends into at least one full limb.
function twos_complement!(t::Memory{Limb}, x::NBig, n::Int)
    lx = nlimbs(x)
    copyto!(t, 1, x.limbs, 1, lx)
    fill!(view(t, lx+1:n), zero(Limb))
    signbit(x) && negate_twos!(t, n)
    return t
end

# Interpret t[1..n] as two's complement (sign limb is all-0 or all-1).
function from_twos_complement!(t::Memory{Limb}, n::Int)
    t[n] >> 63 == 0 && return nbig_from_limbs(1, t, n)
    negate_twos!(t, n)
    return nbig_from_limbs(-1, t, n)
end

for (op, f) in ((:&, :&), (:|, :|), (:xor, :xor))
    @eval function Base.$op(a::NBig, b::NBig)
        n = max(nlimbs(a), nlimbs(b)) + 1
        ta = twos_complement!(Memory{Limb}(undef, n), a, n)
        tb = twos_complement!(Memory{Limb}(undef, n), b, n)
        @inbounds for i in 1:n
            ta[i] = $f(ta[i], tb[i])
        end
        return from_twos_complement!(ta, n)
    end
end

function Base.trailing_zeros(x::NBig)
    iszero(x) && throw(DomainError(x, "trailing_zeros of zero is undefined"))
    i = 1
    @inbounds while x.limbs[i] == 0
        i += 1
    end
    return 64 * (i - 1) + trailing_zeros(@inbounds x.limbs[i])
end

function Base.count_ones(x::NBig)
    signbit(x) && throw(DomainError(x, "count_ones of a negative NBig is undefined"))
    c = 0
    @inbounds for i in 1:nlimbs(x)
        c += count_ones(x.limbs[i])
    end
    return c
end

const DIGIT_CHARS = codeunits("0123456789abcdefghijklmnopqrstuvwxyz")

# --- Subquadratic base conversion (non-power-of-two bases) -----------------
# Both directions recurse by splitting exactly in half around powers
# powtab[i] = bb^(2^(i-1)) of the big base bb = base^k, so each step is one
# NBig divrem!/mul! and inherits Karatsuba + the NTT (O(M(n) log n)).
# Power-of-two bases never reach here — string/parse keep their O(n) bit paths.
# Threshold is in limbs and only coarsely tuned; the win is asymptotic.
const STR_DC_THRESHOLD = 40

# Emit exactly 2^level base-bb chunks of x (0 <= x < powtab[level+1]),
# little-endian, into out. level 0 is the single-limb base case.
function radix_emit!(out::Vector{Limb}, x::NBig, level::Int, powtab::Vector{NBig})
    if level == 0
        push!(out, iszero(x) ? zero(Limb) : (@inbounds x.limbs[1]))
        return
    end
    q, r = divrem(x, @inbounds powtab[level])   # bb^(2^(level-1)), the half point
    radix_emit!(out, r, level - 1, powtab)       # low half
    radix_emit!(out, q, level - 1, powtab)       # high half
    return
end

# Little-endian base-bb chunks of the nonneg NBig x (each < bb). Caller passes
# abs(x); identical output to radix_chunks! but subquadratic.
function radix_chunks_dc(x::NBig, bb::Limb)
    iszero(x) && return Limb[]
    powtab = NBig[NBig(bb)]                       # powtab[i] = bb^(2^(i-1))
    while (@inbounds powtab[end]) <= x
        push!(powtab, powtab[end] * powtab[end])
    end
    out = Limb[]
    radix_emit!(out, x, length(powtab) - 1, powtab)
    while length(out) > 1 && (@inbounds out[end]) == zero(Limb)
        pop!(out)                                 # trim high zero chunks
    end
    return out
end

# Value of little-endian base-bb chunks c[lo .. lo + 2^level - 1] (each < bb).
# Indices past the end of c read as zero, so c need not be padded.
function radix_combine(c::Vector{Limb}, lo::Int, level::Int, powtab::Vector{NBig})
    level == 0 && return NBig(lo <= length(c) ? (@inbounds c[lo]) : zero(Limb))
    h = 1 << (level - 1)
    low = radix_combine(c, lo, level - 1, powtab)
    high = radix_combine(c, lo + h, level - 1, powtab)
    return high * (@inbounds powtab[level]) + low
end

function Base.string(x::NBig; base::Integer = 10, pad::Integer = 1)
    2 <= base <= 36 || throw(ArgumentError("base must be in 2:36, got $base"))
    n = nlimbs(x)
    bb, k = big_base(Int(base))
    if !ispow2(bb) && n >= STR_DC_THRESHOLD
        chunks = radix_chunks_dc(abs(x), bb)
    else
        scratch = Memory{Limb}(undef, n)
        copyto!(scratch, 1, x.limbs, 1, n)
        chunks = radix_chunks!(scratch, n, bb)
    end
    # digit count of the top chunk (chunks may be empty for zero)
    ndig = (length(chunks) - 1) * k
    top = isempty(chunks) ? zero(Limb) : chunks[end]
    while top > 0
        ndig += 1
        top = div(top, base % Limb)
    end
    width = max(ndig, Int(pad), 1)
    neg = signbit(x)
    buf = Base.StringMemory(width + neg)
    j = width + neg
    for (ci, c) in enumerate(chunks)
        lim = ci == length(chunks) ? ndig - (length(chunks) - 1) * k : k
        for _ in 1:lim
            c, d = divrem(c, base % Limb)
            @inbounds buf[j] = DIGIT_CHARS[d+1]
            j -= 1
        end
    end
    @inbounds while j > (neg ? 1 : 0)
        buf[j] = UInt8('0')
        j -= 1
    end
    neg && (@inbounds buf[1] = UInt8('-'))
    return Base.unsafe_takestring(buf)
end

Base.show(io::IO, x::NBig) = print(io, string(x))

function Base.tryparse(::Type{NBig}, s::AbstractString; base::Integer = 10)
    2 <= base <= 36 || throw(ArgumentError("base must be in 2:36, got $base"))
    cs = lstrip(isspace, rstrip(isspace, s))
    isempty(cs) && return nothing
    neg = false
    c1 = cs[1]
    if c1 == '-' || c1 == '+'
        neg = c1 == '-'
        cs = SubString(cs, nextind(cs, 1))
        isempty(cs) && return nothing
    end
    bb, k = big_base(Int(base))
    ndig = length(cs)
    if ispow2(base)
        # Each digit is lg = log2(base) bits at a fixed offset: pack them
        # straight into the limb buffer, most-significant char first. O(n).
        lg = trailing_zeros(Int(base))
        nl = cld(ndig * lg, 64)
        limbs = Memory{Limb}(undef, nl)
        fill!(limbs, zero(Limb))
        p = ndig - 1                       # significance position of current char
        for c in cs
            d = '0' <= c <= '9' ? c - '0' :
                'a' <= c <= 'z' ? c - 'a' + 10 :
                'A' <= c <= 'Z' ? c - 'A' + 10 : 99
            d < base || return nothing
            bitpos = p * lg
            q = bitpos >> 6
            off = bitpos & 63
            @inbounds limbs[q+1] |= (d % Limb) << off
            if off + lg > 64
                @inbounds limbs[q+2] |= (d % Limb) >> (64 - off)
            end
            p -= 1
        end
        x = nbig_from_limbs(1, limbs, nl)
        return neg ? -x : x
    end
    if ndig >= STR_DC_THRESHOLD * k
        # Divide-and-conquer: group digits into base-bb chunks (most-significant
        # group first, possibly short), then combine pairwise through NBig mul!.
        firstlen = ndig % k
        firstlen == 0 && (firstlen = k)
        g = Vector{Limb}(undef, cld(ndig, k))
        gi = 1
        chunk = zero(Limb)
        cnt = 0
        target = firstlen
        for c in cs
            d = '0' <= c <= '9' ? c - '0' :
                'a' <= c <= 'z' ? c - 'a' + 10 :
                'A' <= c <= 'Z' ? c - 'A' + 10 : 99
            d < base || return nothing
            chunk = chunk * (base % Limb) + (d % Limb)
            cnt += 1
            if cnt == target
                @inbounds g[gi] = chunk
                gi += 1
                chunk = zero(Limb)
                cnt = 0
                target = k
            end
        end
        reverse!(g)   # little-endian for radix_combine
        nchunks = length(g)
        level = nchunks <= 1 ? 0 : Base.top_set_bit(nchunks - 1)   # 2^level >= nchunks
        powtab = Vector{NBig}(undef, level)
        level >= 1 && (powtab[1] = NBig(bb))
        for i in 2:level
            @inbounds powtab[i] = powtab[i-1] * powtab[i-1]
        end
        x = radix_combine(g, 1, level, powtab)
        return neg ? -x : x
    end
    bigb = NBig(bb)
    x = NBig(0, EMPTY_LIMBS)
    chunk = zero(Limb)
    nd = 0
    for c in cs
        d = '0' <= c <= '9' ? c - '0' :
            'a' <= c <= 'z' ? c - 'a' + 10 :
            'A' <= c <= 'Z' ? c - 'A' + 10 : 99
        d < base || return nothing
        chunk = chunk * (base % Limb) + (d % Limb)
        nd += 1
        if nd == k
            x = x * bigb + NBig(chunk)
            chunk = zero(Limb)
            nd = 0
        end
    end
    if nd > 0
        x = x * NBig(Limb(base)^nd) + NBig(chunk)
    end
    return neg ? -x : x
end

function Base.parse(::Type{NBig}, s::AbstractString; base::Integer = 10)
    r = tryparse(NBig, s; base)
    r === nothing && throw(ArgumentError("invalid base-$base digit string: $(repr(s))"))
    return r
end

# Delegating keeps isequal/hash consistent with Int and BigInt; a native
# limb-walking hash is a post-v1 optimization.
Base.hash(x::NBig, h::UInt) = hash(BigInt(x), h)

Base.Float64(x::NBig) = Float64(BigInt(x))
Base.AbstractFloat(x::NBig) = Float64(x)

Base.zero(::Type{NBig}) = NBig(0, EMPTY_LIMBS)
Base.one(::Type{NBig}) = NBig(1)

Base.promote_rule(::Type{NBig}, ::Type{<:Integer}) = NBig

# --- NBig ↔ BitInteger64 fast paths -----------------------------------------
# A BitInteger64 operand is a single limb of magnitude, so these route through
# the _1! kernels instead of promoting (which allocates a temp NBig and runs
# the full n-limb paths). 128-bit ints keep the promotion path.

const BitInt64 = Base.BitInteger64

# (sign, magnitude-as-Limb); flipsign on typemin(Int64) wraps to typemin,
# whose unsigned reinterpretation 0x80… is the correct magnitude.
@inline limb_split(b::Base.BitSigned64) = (Int(sign(b)), unsigned(flipsign(Int64(b), b)) % Limb)
@inline limb_split(b::Base.BitUnsigned64) = (Int(!iszero(b)), Limb(b))

@inline function nbig_from_limb(sgn::Int, mag::Limb)
    (sgn == 0 || mag == 0) && return NBig(0, EMPTY_LIMBS)
    m = Memory{Limb}(undef, 1)
    @inbounds m[1] = mag
    return NBig(sgn, m)
end

function add_small(a::NBig, s::Int, mag::Limb)
    s == 0 && return a
    iszero(a) && return nbig_from_limb(s, mag)
    la = nlimbs(a)
    if sign(a) == s
        r = Memory{Limb}(undef, la + 1)
        copyto!(r, 1, a.limbs, 1, la)
        @inbounds r[la+1] = 0
        add_carry!(r, 0, la + 1, 1, mag)
        return nbig_from_limbs(s, r, la + 1)
    elseif la == 1
        al = @inbounds a.limbs[1]
        return al >= mag ? nbig_from_limb(sign(a), al - mag) : nbig_from_limb(s, mag - al)
    else
        r = Memory{Limb}(undef, la)
        sub_1!(r, 0, a.limbs, 0, la, mag)
        return nbig_from_limbs(sign(a), r, la)
    end
end

Base.:+(a::NBig, b::BitInt64) = ((s, mag) = limb_split(b); add_small(a, s, mag))
Base.:+(b::BitInt64, a::NBig) = a + b
Base.:-(a::NBig, b::BitInt64) = ((s, mag) = limb_split(b); add_small(a, -s, mag))
Base.:-(b::BitInt64, a::NBig) = ((s, mag) = limb_split(b); add_small(-a, s, mag))

function Base.:*(a::NBig, b::BitInt64)
    s, mag = limb_split(b)
    (iszero(a) || s == 0) && return NBig(0, EMPTY_LIMBS)
    la = nlimbs(a)
    r = Memory{Limb}(undef, la + 1)
    @inbounds r[la+1] = mul_1!(r, 0, a.limbs, 0, la, mag)
    return nbig_from_limbs(sign(a) * s, r, la + 1)
end
Base.:*(b::BitInt64, a::NBig) = a * b

function Base.cmp(a::NBig, b::BitInt64)
    s, mag = limb_split(b)
    sa = sign(a)
    sa != s && return sa < s ? -1 : 1
    sa == 0 && return 0
    al = @inbounds a.limbs[1]
    c = nlimbs(a) > 1 ? 1 : (al == mag ? 0 : (al < mag ? -1 : 1))
    return sa < 0 ? -c : c
end
Base.cmp(b::BitInt64, a::NBig) = -cmp(a, b)
Base.:(==)(a::NBig, b::BitInt64) = cmp(a, b) == 0
Base.:(==)(b::BitInt64, a::NBig) = cmp(a, b) == 0
Base.:<(a::NBig, b::BitInt64) = cmp(a, b) < 0
Base.:<(b::BitInt64, a::NBig) = cmp(a, b) > 0
Base.:<=(a::NBig, b::BitInt64) = cmp(a, b) <= 0
Base.:<=(b::BitInt64, a::NBig) = cmp(a, b) >= 0
Base.isless(a::NBig, b::BitInt64) = cmp(a, b) < 0
Base.isless(b::BitInt64, a::NBig) = cmp(a, b) > 0

# Small divisor: one divrem_1! pass on the magnitude, then sign fixups.
function Base.divrem(a::NBig, b::BitInt64)
    s, mag = limb_split(b)
    s == 0 && throw(DivideError())
    iszero(a) && return (NBig(0, EMPTY_LIMBS), NBig(0, EMPTY_LIMBS))
    la = nlimbs(a)
    q = Memory{Limb}(undef, la)
    r = divrem_1!(q, 0, a.limbs, 0, la, mag)
    return (nbig_from_limbs(sign(a) * s, q, la), nbig_from_limb(sign(a), r))
end

# Small dividend: |b| is one limb, so the quotient is 0 unless a is too.
function Base.divrem(b::BitInt64, a::NBig)
    iszero(a) && throw(DivideError())
    s, mag = limb_split(b)
    s == 0 && return (NBig(0, EMPTY_LIMBS), NBig(0, EMPTY_LIMBS))
    al = @inbounds a.limbs[1]
    (nlimbs(a) > 1 || al > mag) && return (NBig(0, EMPTY_LIMBS), nbig_from_limb(s, mag))
    qm, rm = divrem(mag, al)
    return (nbig_from_limb(s * sign(a), qm), nbig_from_limb(s, rm))
end

for (A, B) in ((:NBig, :BitInt64), (:BitInt64, :NBig))
    @eval begin
        Base.div(a::$A, b::$B) = divrem(a, b)[1]
        Base.rem(a::$A, b::$B) = divrem(a, b)[2]
        function Base.mod(a::$A, b::$B)
            r = rem(a, b)
            return (iszero(r) || sign(r) == sign(b)) ? r : r + b
        end
        function Base.fld(a::$A, b::$B)
            q, r = divrem(a, b)
            return (!iszero(r) && sign(r) != sign(b)) ? q - 1 : q
        end
        function Base.cld(a::$A, b::$B)
            q, r = divrem(a, b)
            return (!iszero(r) && sign(r) == sign(b)) ? q + 1 : q
        end
    end
end

function Base.gcd(a::NBig, b::BitInt64)
    s, mag = limb_split(b)
    s == 0 && return abs(a)
    iszero(a) && return nbig_from_limb(1, mag)
    la = nlimbs(a)
    q = Memory{Limb}(undef, la)
    r = divrem_1!(q, 0, a.limbs, 0, la, mag)
    return nbig_from_limb(1, gcd(r, mag))
end
Base.gcd(b::BitInt64, a::NBig) = gcd(a, b)

# mod(a, m) follows m's sign, so the residue is exactly representable in
# typeof(m); the whole inverse then runs in native arithmetic.
function Base.invmod(a::NBig, m::T) where {T<:BitInt64}
    iszero(m) && throw(DomainError(m, "`m` must be nonzero."))
    r = mod(a, m)
    rl = iszero(r) ? zero(Limb) : @inbounds r.limbs[1]
    b = T <: Unsigned ? rl % T : flipsign(rl % Int64, r.signlen) % T
    return invmod(b, m)
end
Base.invmod(b::BitInt64, m::NBig) = invmod(NBig(b), m)

# NBig overloads of the powermod exponent-bit accessors (algorithms.jl).
@inline expbit(n::NBig, j::Int) = ((@inbounds n.limbs[(j >>> 6) + 1]) >> (j & 63)) % Bool
@inline expbits(n::NBig) = Base.top_set_bit(n)

# Reduce the base to a native int (mod(a, m) follows m's sign, so it is
# exactly representable in typeof(m)), then square-and-multiply in native
# arithmetic: every product of residues fits the widemul type.
function Base.powermod(a::NBig, n::Integer, m::T) where {T<:BitInt64}
    iszero(m) && throw(DivideError())
    r = mod(a, m)
    rl = iszero(r) ? zero(Limb) : @inbounds r.limbs[1]
    b = T <: Unsigned ? rl % T : flipsign(rl % Int64, r.signlen) % T
    if signbit(n)
        # Base handles negative native exponents; NBig exponents invert here
        # and re-enter this method with -n (no overflow risk for NBig).
        n isa NBig || return powermod(b, n, m)
        return powermod(NBig(invmod(b, m)), -n, m)
    end
    res = mod(one(T), m)  # covers n == 0 and m == ±1
    iszero(n) && return res
    for j in (expbits(n)-1):-1:0
        res = mod(widemul(res, res), m) % T
        if expbit(n, j)
            res = mod(widemul(res, b), m) % T
        end
    end
    return res
end

Base.widen(::Type{NBig}) = NBig

function Base.top_set_bit(x::NBig)
    n = nlimbs(x)
    n == 0 && return 0
    return magnitude_bits(x.limbs, 0, n)
end

# Arbitrary precision never overflows (matches Base's BigInt methods).
Base.Checked.add_with_overflow(a::NBig, b::NBig) = a + b, false
Base.Checked.sub_with_overflow(a::NBig, b::NBig) = a - b, false
Base.Checked.mul_with_overflow(a::NBig, b::NBig) = a * b, false

# Base's generic powermod assumes fixed-width tricks (unsigned/prevpow);
# route mixed calls with an NBig modulus to the Montgomery implementation.
Base.powermod(a::Integer, n::Integer, m::NBig) = powermod(NBig(a), n, m)

Base.rem(x::NBig, ::Type{Bool}) = isodd(x)

# Two's-complement truncation to any native width (BigInt-compatible x % T).
function Base.rem(x::NBig, ::Type{T}) where {T<:Union{Base.BitUnsigned, Base.BitSigned}}
    u = zero(T)
    for l in 1:min(nlimbs(x), cld(sizeof(T), 8))
        u += (@inbounds x.limbs[l] % T) << (64 * (l - 1))
    end
    return flipsign(u, x.signlen)
end

# Checked constructors, mirroring Base's BigInt versions: sub-limb types
# go through the checked Limb-width conversion, wider ones bound the limb
# count so x % T is exact (with a sign-consistency check for signed T).
function (::Type{T})(x::NBig) where {T<:Base.BitUnsigned}
    if sizeof(T) < sizeof(Limb)
        return convert(T, convert(Limb, x))
    else
        0 <= x.signlen <= cld(sizeof(T), sizeof(Limb)) || throw(InexactError(nameof(T), T, x))
        return x % T
    end
end

function (::Type{T})(x::NBig) where {T<:Base.BitSigned}
    if sizeof(T) < sizeof(Limb)
        return convert(T, convert(Int64, x))
    else
        nlimbs(x) <= cld(sizeof(T), sizeof(Limb)) || throw(InexactError(nameof(T), T, x))
        y = x % T
        (x.signlen > 0) ⊻ (y > 0) && throw(InexactError(nameof(T), T, x))  # catch overflow
        return y
    end
end
