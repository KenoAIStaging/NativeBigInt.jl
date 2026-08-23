# Multiplication and squaring above the basecase kernels: benchmark-tuned
# dispatch thresholds, subtractive Karatsuba, and the mul!/sqr! entry points
# that route between them and the fp NTT (fpntt.jl).  Toom-3 used to sit
# between Karatsuba and the NTT, but the fp NTT squeezed its winning band to
# 240-340 limbs (peak edge ~20% at 240, ~7% by 320) and it was deleted in
# favor of Karatsuba stretching to the NTT crossover.

# Benchmark-tuned dispatch thresholds for mul!/sqr!, in limbs.
# Basecase → Karatsuba (bench/bench_kar_thr.jl, bench/bench_sqr.jl):
# sqr_basecase! does half the multiplies of mul_basecase!, so squaring stays
# basecase longer.
const MUL_KARATSUBA_THRESHOLD = 29
const SQR_KARATSUBA_THRESHOLD = 52
# Karatsuba → two-prime fp NTT.  The two-prime engine wins over single-prime at
# every size above this line (its transforms are 2/2.5 the points since b ≈ 43
# vs ≈ 17-24; only linear overhead ever favored single-prime), so dispatch is
# fp2-only.
#
# mul! selects on m + n, and a 2D (m, n) sweep confirms that shape: from 1:1 to
# 8:1 the crossover sits between 240 total limbs (classical wins everywhere,
# 1.01-1.09) and 248-256 (NTT wins or ties, 0.83-1.00) — hence 124.  Both sides
# there pay their per-call scratch malloc as mul! does; hoisting Karatsuba's out
# flatters it enough to move this constant by ~25 limbs.
# For unbalanced operands the NTT needs the smaller one substantial (the
# chunked Karatsuba path is ~max·min^0.585 while the NTT pays for the
# combined length); at min = 64 the admitted region wins throughout
# (0.74-0.91 at n = 64 across m = 512..8192; n = 48 loses out to 8:1).
# Squaring crosses later (~152: sqr_kar! saves half the classical work while
# the NTT saves only one of its three packs), so its cut stays at 160.
const MUL_FPNTT_MIN = 64         # smaller operand at least this many limbs
const MUL_FPNTT_THRESHOLD = 124  # average operand at least this many limbs
const SQR_FPNTT_THRESHOLD = 160  # operand at least this many limbs

# Value comparison of la-limb a vs lb-limb b (la >= lb): strip a's zero top
# limbs (split halves are zero-padded, cmp_limbs trusts lengths) and delegate.
@inline function cmp_padded(a::Memory{Limb}, ao::Int, la::Int, b::Memory{Limb}, bo::Int, lb::Int)
    @inbounds while la > lb && a[ao+la] == 0
        la -= 1
    end
    return cmp_limbs(a, ao, la, b, bo, lb)
end

# d[1..lo_len] = |x_lo - x_hi| where x_lo = x[xo+1 .. xo+lo_len] and
# x_hi = x[xo+lo_len+1 .. xo+lo_len+hi_len], hi_len <= lo_len.
# Returns true iff the difference is negative (x_lo < x_hi).
function abs_diff!(d::Memory{Limb}, dof::Int, x::Memory{Limb}, xo::Int, lo_len::Int, hi_len::Int)
    if cmp_padded(x, xo, lo_len, x, xo + lo_len, hi_len) >= 0
        sub!(d, dof, x, xo, lo_len, x, xo + lo_len, hi_len)
        return false
    else
        # x_lo < x_hi < B^hi_len forces x_lo's limbs above hi_len to be zero
        sub_n!(d, dof, x, xo + lo_len, x, xo, hi_len)
        fill!(view(d, dof+hi_len+1:dof+lo_len), zero(Limb))
        return true
    end
end

# Scratch limbs mul_kar! needs for an n x n product: 4*ceil(n/2) per level.
function kar_scratch_len(n::Int, thr::Int=MUL_KARATSUBA_THRESHOLD)
    len = 0
    while n >= thr
        n = (n + 1) >> 1
        len += 4n
    end
    return len
end

# Balanced n x n subtractive Karatsuba: r[1..2n] = a[1..n] * b[1..n].
# a*b = hi*B^(2h2) + (lo + hi - s*mid)*B^h2 + lo where mid = |a_lo-a_hi|*|b_lo-b_hi|.
# The middle term equals a_lo*b_hi + a_hi*b_lo >= 0, so carries never underflow.
# scratch must have kar_scratch_len(n) limbs free at so; r must not alias a/b.
function mul_kar!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int,
                  b::Memory{Limb}, bo::Int, n::Int, scratch::Memory{Limb}, so::Int,
                  thr::Int=MUL_KARATSUBA_THRESHOLD)
    if n < thr
        return mul_basecase!(r, ro, a, ao, n, b, bo, n)
    end
    h2 = (n + 1) >> 1   # low-half length
    h = n - h2          # high-half length (h2 or h2-1)
    L = 2h2
    mid = so            # L limbs: |a_lo-a_hi| * |b_lo-b_hi|
    tmp = so + L        # L limbs: the two differences, then lo+hi±mid
    rec = so + 2L       # recursion scratch
    nega = abs_diff!(scratch, tmp, a, ao, h2, h)
    negb = abs_diff!(scratch, tmp + h2, b, bo, h2, h)
    mul_kar!(scratch, mid, scratch, tmp, scratch, tmp + h2, h2, scratch, rec, thr)
    mul_kar!(r, ro, a, ao, b, bo, h2, scratch, rec, thr)                 # lo
    mul_kar!(r, ro + L, a, ao + h2, b, bo + h2, h, scratch, rec, thr)    # hi
    c = add!(scratch, tmp, r, ro, L, r, ro + L, 2h)                 # tmp = lo + hi
    if nega == negb
        c -= sub_n!(scratch, tmp, scratch, tmp, scratch, mid, L)
    else
        c += add_n!(scratch, tmp, scratch, tmp, scratch, mid, L)
    end
    add_into!(r, ro + h2, 2n - h2, scratch, tmp, L)
    add_carry!(r, ro + h2, 2n - h2, L + 1, c)
    return nothing
end

# Balanced Karatsuba squaring: r[1..2n] = a[1..n]^2. Same recursion shape as
# mul_kar! with mid = (a_lo - a_hi)^2 >= 0, so the middle term is always
# lo + hi - mid and there is no sign tracking. Scratch layout matches
# kar_scratch_len(n); r must not alias a.
function sqr_kar!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, n::Int,
                  scratch::Memory{Limb}, so::Int, thr::Int=SQR_KARATSUBA_THRESHOLD)
    if n < thr
        return sqr_basecase!(r, ro, a, ao, n)
    end
    h2 = (n + 1) >> 1   # low-half length
    h = n - h2          # high-half length (h2 or h2-1)
    L = 2h2
    mid = so            # L limbs: (a_lo - a_hi)^2
    tmp = so + L        # L limbs: the difference, then lo + hi - mid
    rec = so + 2L       # recursion scratch
    abs_diff!(scratch, tmp, a, ao, h2, h)
    sqr_kar!(scratch, mid, scratch, tmp, h2, scratch, rec, thr)
    sqr_kar!(r, ro, a, ao, h2, scratch, rec, thr)                 # lo
    sqr_kar!(r, ro + L, a, ao + h2, h, scratch, rec, thr)         # hi
    c = add!(scratch, tmp, r, ro, L, r, ro + L, 2h)               # tmp = lo + hi
    c -= sub_n!(scratch, tmp, scratch, tmp, scratch, mid, L)
    add_into!(r, ro + h2, 2n - h2, scratch, tmp, L)
    add_carry!(r, ro + h2, 2n - h2, L + 1, c)
    return nothing
end

# Scratch limbs the balanced dispatchers need for an n x n product/square
# (Karatsuba all the way up to the NTT crossover).
mul_scratch_len(n::Int) = kar_scratch_len(n)
sqr_scratch_len(n::Int) = kar_scratch_len(n, SQR_KARATSUBA_THRESHOLD)

# Squaring dispatch with caller-provided scratch (sqr_scratch_len(n) limbs
# at so); squaring is inherently balanced, so this is sqr! itself.
function sqr!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, n::Int,
              scratch::Memory{Limb}, so::Int)
    if n < SQR_KARATSUBA_THRESHOLD
        sqr_basecase!(r, ro, a, ao, n)
    elseif n < SQR_FPNTT_THRESHOLD
        sqr_kar!(r, ro, a, ao, n, scratch, so)
    else
        sqr_fpntt2!(r, ro, a, ao, n)
    end
    return nothing
end

# Allocating convenience form: r[1..2n] = a[1..n]^2, n >= 1; r must not
# alias a. Basecase/NTT sizes skip the scratch allocation entirely.
function sqr!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, n::Int)
    if n < SQR_KARATSUBA_THRESHOLD
        return sqr_basecase!(r, ro, a, ao, n)
    end
    if n >= SQR_FPNTT_THRESHOLD
        return sqr_fpntt2!(r, ro, a, ao, n)
    end
    sqr!(r, ro, a, ao, n, Memory{Limb}(undef, sqr_scratch_len(n)), 0)
    return nothing
end

# Balanced n x n dispatch with caller-provided scratch (mul_scratch_len(n)
# limbs at so; the NTT branch allocates its own and ignores it).
function mul_bal!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int,
                  b::Memory{Limb}, bo::Int, n::Int, scratch::Memory{Limb}, so::Int)
    if n < MUL_KARATSUBA_THRESHOLD
        mul_basecase!(r, ro, a, ao, n, b, bo, n)
    elseif n < MUL_FPNTT_THRESHOLD
        mul_kar!(r, ro, a, ao, b, bo, n, scratch, so)
    else
        mul_fpntt2!(r, ro, a, ao, n, b, bo, n)
    end
    return nothing
end

# General product r[1..m+n] = a[1..m] * b[1..n], m >= n >= 1; r must not
# alias a or b. Balanced Karatsuba on n-limb chunks of a; each block's low n limbs
# accumulate into r, its high limbs land in fresh territory (plus carry).
function mul!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, m::Int,
              b::Memory{Limb}, bo::Int, n::Int)
    if n < MUL_KARATSUBA_THRESHOLD
        return mul_basecase!(r, ro, a, ao, m, b, bo, n)
    end
    if n >= MUL_FPNTT_MIN && m + n >= 2MUL_FPNTT_THRESHOLD
        return mul_fpntt2!(r, ro, a, ao, m, b, bo, n)
    end
    scratch = Memory{Limb}(undef, 2n + mul_scratch_len(n))
    mul_bal!(r, ro, a, ao, b, bo, n, scratch, 2n)
    i = n
    while i < m
        chunk = min(n, m - i)
        if chunk == n
            mul_bal!(scratch, 0, a, ao + i, b, bo, n, scratch, 2n)
        else
            mul!(scratch, 0, b, bo, n, a, ao + i, chunk)  # ragged tail, n x chunk
        end
        c = add_n!(r, ro + i, r, ro + i, scratch, 0, n)
        copy_tail!(r, ro + i, scratch, 0, n + 1, n + chunk)
        add_carry!(r, ro + i, m + n - i, n + 1, c)
        i += chunk
    end
    return nothing
end
