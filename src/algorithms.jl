# Multi-limb algorithms built on the kernels: powermod and radix conversion.
# (Multiplication lives in mul.jl, division in div.jl, Montgomery reduction
# in montgomery.jl, the Lehmer gcds in gcd.jl, Karatsuba sqrt in sqrt.jl.)

# O(1) exponent bit access; NBig overloads live in nbig.jl.
@inline expbit(e::Integer, i::Int) = (e >>> i) % Bool
@inline expbits(e::Integer) = Base.top_set_bit(e)

# b^e mod m on magnitudes: m has k limbs (m[k] ≠ 0, m > 1), 0 < b < m
# (lb limbs), e > 0 any Integer supporting expbit/expbits. Returns a k-limb
# Memory (unnormalized). Sliding-window exponentiation; for odd m the values
# live in Montgomery form with redc! after each mul/sqr, for even m each
# product is reduced with divrem! instead. Above the per-parity Barrett
# thresholds both give way to plain-domain Barrett reduction (mu_reduce!, which
# rides mul!'s subquadratic engines and amortizes one invertappr! reciprocal
# across every product here). The flag is overridable for threshold
# benchmarks/tests.
function powermod_limbs(b::Memory{Limb}, lb::Int, e::Integer,
                        m::Memory{Limb}, k::Int,
                        barrett::Bool = k >= (isodd(@inbounds m[1]) ?
                                              BARRETT_THRESHOLD : BARRETT_EVEN_THRESHOLD))
    odd = !barrett && isodd(@inbounds m[1])
    ninv = odd ? mont_ninv(@inbounds m[1]) : zero(Limb)
    mured = barrett ? mu_reduce_setup(m, 0, k) : mu_reduce_empty()
    nbits = expbits(e)
    w = nbits <= 8 ? 1 : nbits <= 24 ? 2 : nbits <= 80 ? 3 : nbits <= 240 ? 4 : 5
    tsize = 1 << (w - 1)   # table of odd powers b^1, b^3, …, b^(2^w - 1)
    acc = Memory{Limb}(undef, k)
    prod = Memory{Limb}(undef, 2k + 1)
    qbuf = Memory{Limb}(undef, k + 2)
    table = Memory{Limb}(undef, tsize * k)

    # dst[1..k] = x*y (or x^2) brought back into the working domain
    function mulred!(dst::Memory{Limb}, dsto::Int, x::Memory{Limb}, xo::Int,
                     y::Memory{Limb}, yo::Int)
        if x === y && xo == yo
            sqr!(prod, 0, x, xo, k)
        else
            mul!(prod, 0, x, xo, k, y, yo, k)
        end
        @inbounds prod[2k+1] = 0
        if barrett
            # x, y < m so T < m² ≤ m·β^k, which is mu_reduce!'s precondition
            mu_reduce!(dst, dsto, prod, 0, k, mured)
        elseif odd
            redc!(dst, dsto, prod, 0, m, 0, k, ninv)
        else
            lt = normlen(prod, 0, 2k)
            if lt < k || cmp_limbs(prod, 0, lt, m, 0, k) < 0
                copyto!(dst, dsto + 1, prod, 1, lt)
                fill!(view(dst, dsto+lt+1:dsto+k), zero(Limb))
            else
                divrem!(qbuf, 0, dst, dsto, prod, 0, lt, m, 0, k)
            end
        end
    end

    # table slot 0 = base: b·β^k mod m in Montgomery form, else b zero-padded
    if odd
        fill!(view(prod, 1:k), zero(Limb))
        copyto!(prod, k + 1, b, 1, lb)
        divrem!(qbuf, 0, table, 0, prod, 0, k + lb, m, 0, k)
    else
        copyto!(table, 1, b, 1, lb)
        fill!(view(table, lb+1:k), zero(Limb))
    end
    if tsize > 1
        bsq = Memory{Limb}(undef, k)
        mulred!(bsq, 0, table, 0, table, 0)
        for j in 1:tsize-1
            mulred!(table, j * k, table, (j - 1) * k, bsq, 0)
        end
    end

    first = true
    i = nbits - 1
    while i >= 0
        if !expbit(e, i)
            first || mulred!(acc, 0, acc, 0, acc, 0)
            i -= 1
        else
            l = min(w, i + 1)
            while !expbit(e, i - l + 1)   # window must end on a set bit
                l -= 1
            end
            val = 0
            for j in 0:l-1
                val = (val << 1) | Int(expbit(e, i - j))
            end
            if first
                copyto!(acc, 1, table, (val >> 1) * k + 1, k)
                first = false
            else
                for _ in 1:l
                    mulred!(acc, 0, acc, 0, acc, 0)
                end
                mulred!(acc, 0, acc, 0, table, (val >> 1) * k)
            end
            i -= l
        end
    end
    if odd   # leave Montgomery form: one reduction of the bare value
        copyto!(prod, 1, acc, 1, k)
        fill!(view(prod, k+1:2k+1), zero(Limb))
        redc!(acc, 0, prod, 0, m, 0, k, ninv)
    end
    return acc
end

# Largest power of base that fits in a limb: (base^k, k).
function big_base(base::Int)
    bb = Limb(base)
    k = 1
    while true
        nb, ovf = Base.mul_with_overflow(bb, Limb(base))
        ovf && return bb, k
        bb, k = nb, k + 1
    end
end

# Little-endian base^k digit chunks of the n-limb magnitude in a (destroyed:
# a is divided down in place). Repeated divrem_1! — O(n²), fine for v1;
# divide-and-conquer is a post-v1 extension for very large n. A power-of-two
# base^k = 2^c needs no division: each chunk is a fixed-width c-bit window (O(n)).
function radix_chunks!(a::Memory{Limb}, n::Int, bb::Limb)
    ispow2(bb) && return radix_chunks_pow2(a, n, trailing_zeros(bb))
    chunks = Limb[]
    while n > 0
        push!(chunks, divrem_1!(a, 0, a, 0, n, bb))
        n = normlen(a, 0, n)
    end
    return chunks
end

# Little-endian c-bit windows of the n-limb magnitude (a[n] ≠ 0, c ≤ 63):
# identical output to repeatedly dividing by 2^c, but O(n). Does not touch a.
function radix_chunks_pow2(a::Memory{Limb}, n::Int, c::Int)
    n == 0 && return Limb[]
    bitlen = magnitude_bits(a, 0, n)
    nchunks = cld(bitlen, c)
    chunks = Vector{Limb}(undef, nchunks)
    mask = (one(Limb) << c) - one(Limb)
    @inbounds for t in 0:nchunks-1
        b = t * c
        q = b >> 6
        off = b & 63
        w = a[q+1] >> off
        if off != 0 && q+2 <= n   # window straddles a limb boundary
            w |= a[q+2] << (64 - off)
        end
        chunks[t+1] = w & mask
    end
    return chunks
end

