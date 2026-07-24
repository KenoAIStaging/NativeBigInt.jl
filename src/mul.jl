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

# --- Low short products (Mulders): r[1..k] = product mod β^k -----------------
#
# "Short" refers to the skipped high-half work; the low k limbs are exact.
# Contract shared by mullo!/sqrlo! and their basecases: r needs k+2 limbs of
# capacity at ro and r[ro+k+1..ro+k+2] are clobbered — the slack lets the
# truncated basecases keep the paired addmul_2! rows (which write, not add,
# their two top limbs) and lets near-full products run in place.

# Basecase → Mulders recursion crossover and the Mulders split point
# (bench/bench_mullo_thr.jl): the paired-row truncated basecases hold ~0.55-
# 0.75x of the full product deep into the Karatsuba range, so the recursion
# only wins once its full sub-product is solidly Karatsuba. Above the FULL
# thresholds the balanced full product rides the NTT and a short product
# stops paying: mullo's band closes right at the NTT crossover (~224), while
# sqrlo's cheaper cross term keeps a noisy edge to ~448.
const MULLO_BASECASE_THRESHOLD = 140
const SQRLO_BASECASE_THRESHOLD = 224
const MULLO_FULL_THRESHOLD = 224
const SQRLO_FULL_THRESHOLD = 448
mullo_split(k::Int) = (11 * k) >> 4   # keep ~0.69k in the full low product

# Scratch limbs mullo!/sqrlo! need at so (0 when every path is in-place).
function mullo_scratch_len(k::Int)
    k < MULLO_BASECASE_THRESHOLD && return 0
    k >= MULLO_FULL_THRESHOLD && return 2k        # full product, low k kept
    m = mullo_split(k)
    kk = k - m
    return max(2m, kk + 2 + mullo_scratch_len(kk))
end
function sqrlo_scratch_len(k::Int)
    k < SQRLO_BASECASE_THRESHOLD && return 0
    k >= SQRLO_FULL_THRESHOLD && return 2k
    m = mullo_split(k)
    kk = k - m
    return max(2m + sqr_scratch_len(m), kk + 2 + mullo_scratch_len(kk))
end

# Truncated schoolbook: r[1..k] = a*b mod β^k with paired addmul_2! rows
# clipped at column k; clipped pairs park their two written top limbs in the
# clobber slack. Preconditions (dispatcher-enforced): 2 <= lb <= la <= k and
# la + lb > k + 2, so every unclipped pair's top writes stay inside k+2 and
# the final unpaired row's carry always lands above column k (discarded).
function mullo_basecase!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, la::Int,
                         b::Memory{Limb}, bo::Int, lb::Int, k::Int)
    @inbounds begin
        mul_2!(r, ro, a, ao, la, b[bo+1], b[bo+2])
        j = 3
        while j + 1 <= lb
            len = min(la, k - j + 1)
            addmul_2!(r, ro + j - 1, a, ao, len, b[bo+j], b[bo+j+1])
            j += 2
        end
        if j <= lb
            addmul_1!(r, ro + j - 1, a, ao, min(la, k - j + 1), b[bo+j])
        end
    end
    return nothing
end

# Truncated squaring basecase: off-diagonal triangle rows clipped at column k,
# then sqr_basecase!'s fused double-and-add-diagonal pass over the low k
# columns (an odd k's top slot spills its high limb into the clobber slack).
# Preconditions (dispatcher-enforced): 3 <= n <= k <= 2n - 3.
function sqrlo_basecase!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, n::Int, k::Int)
    @inbounds begin
        len1 = min(n - 2, k - 2)
        mul_2!(r, ro + 2, a, ao + 2, len1, a[ao+1], a[ao+2])
        p = widemul(a[ao+1], a[ao+2])
        r[ro+2] = p % Limb
        t = UInt128(r[ro+3]) + (p >> 64) % Limb
        r[ro+3] = t % Limb
        c = (t >> 64) % Limb
        # carries beyond a clipped row's cap belong to columns > k: drop them
        kpos = 4
        cap = min(k, len1 + 4)
        while c != zero(Limb) && kpos <= cap
            t = UInt128(r[ro+kpos]) + c
            r[ro+kpos] = t % Limb
            c = (t >> 64) % Limb
            kpos += 1
        end
        j = 3
        while j <= n - 2 && 2j < k
            len = min(n - j - 1, k - 2j)
            addmul_2!(r, ro + 2j, a, ao + j + 1, len, a[ao+j], a[ao+j+1])
            p = widemul(a[ao+j], a[ao+j+1])
            t = UInt128(r[ro+2j]) + p % Limb
            r[ro+2j] = t % Limb
            c = ((t >> 64) % Limb) + (p >> 64) % Limb
            kpos = 2j + 1
            cap = min(k, 2j + len + 2)
            while c != zero(Limb) && kpos <= cap
                t = UInt128(r[ro+kpos]) + c
                r[ro+kpos] = t % Limb
                c = (t >> 64) % Limb
                kpos += 1
            end
            j += 2
        end
        if j <= n - 1 && 2j == k
            # the clipped-out pair's split product still owns column k
            r[ro+k] = r[ro+k] + widemul(a[ao+j], a[ao+j+1]) % Limb
        end
        r[ro+1] = zero(Limb)
        bit = zero(Limb)
        cc = zero(Limb)
        for i in 1:((k + 1) >> 1)
            u_lo = r[ro+2i-1]
            u_hi = r[ro+2i]
            d_lo = (u_lo << 1) | bit
            d_hi = (u_hi << 1) | (u_lo >> 63)
            bit = u_hi >> 63
            p = widemul(a[ao+i], a[ao+i])
            t = UInt128(d_lo) + (p % Limb) + cc
            t2 = (t >> 64) + ((p >> 64) % Limb) + d_hi
            r[ro+2i-1] = t % Limb
            r[ro+2i] = t2 % Limb
            cc = (t2 >> 64) % Limb
        end
    end
    return nothing
end

# r[1..k] = a*b mod β^k, la, lb >= 1, k >= 1; r needs k+2 limbs of capacity
# (top two clobbered) and must not alias a/b. Mulders structure: a full
# product of the low ~0.69k blocks plus two recursive cross mullos of the
# remaining ~0.31k columns; near-full and NTT-regime shapes fall back to
# plain mul!. scratch needs mullo_scratch_len(k) limbs at so (allocated when
# not passed).
function mullo!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, la::Int,
                b::Memory{Limb}, bo::Int, lb::Int, k::Int,
                scratch::Union{Memory{Limb},Nothing}=nothing, so::Int=0)
    la > k && (la = k)                 # limbs above β^k cannot reach r
    lb > k && (lb = k)
    if la < lb
        a, ao, la, b, bo, lb = b, bo, lb, a, ao, la
    end
    if la + lb <= k + 2
        # the full product fits r's k+2 capacity: compute it in place
        mul!(r, ro, a, ao, la, b, bo, lb)
        la + lb < k && fill!(view(r, ro+la+lb+1:ro+k), zero(Limb))
        return nothing
    end
    if k < MULLO_BASECASE_THRESHOLD || lb < MUL_KARATSUBA_THRESHOLD
        return mullo_basecase!(r, ro, a, ao, la, b, bo, lb, k)
    end
    if scratch === nothing
        scratch = Memory{Limb}(undef, mullo_scratch_len(k))
        so = 0
    end
    if k >= MULLO_FULL_THRESHOLD
        mul!(scratch, so, a, ao, la, b, bo, lb)
        copyto!(r, ro + 1, scratch, so + 1, k)
        return nothing
    end
    m = mullo_split(k)
    kk = k - m
    lla = min(la, m)
    llb = min(lb, m)
    mul!(scratch, so, a, ao, lla, b, bo, llb)
    plen = lla + llb
    copyto!(r, ro + 1, scratch, so + 1, min(plen, k))
    plen < k && fill!(view(r, ro+plen+1:ro+k), zero(Limb))
    if la > m   # a_hi × b_lo cross term, columns m+1..k (carry-out discarded)
        mullo!(scratch, so, a, ao + m, la - m, b, bo, min(lb, kk), kk,
               scratch, so + kk + 2)
        add_n!(r, ro + m, r, ro + m, scratch, so, kk)
    end
    if lb > m
        mullo!(scratch, so, b, bo + m, lb - m, a, ao, min(la, kk), kk,
               scratch, so + kk + 2)
        add_n!(r, ro + m, r, ro + m, scratch, so, kk)
    end
    return nothing
end

# r[1..k] = a^2 mod β^k, la >= 1, k >= 1; same capacity/clobber contract as
# mullo!. The cross term is one mullo of the two chunks added in twice; the
# a_hi^2 block starts at column 2m >= k and vanishes entirely. scratch needs
# sqrlo_scratch_len(k) limbs at so.
function sqrlo!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, la::Int, k::Int,
                scratch::Union{Memory{Limb},Nothing}=nothing, so::Int=0)
    la > k && (la = k)
    if 2la <= k + 2
        sqr!(r, ro, a, ao, la)
        2la < k && fill!(view(r, ro+2la+1:ro+k), zero(Limb))
        return nothing
    end
    if k < SQRLO_BASECASE_THRESHOLD
        return sqrlo_basecase!(r, ro, a, ao, la, k)
    end
    if scratch === nothing
        scratch = Memory{Limb}(undef, sqrlo_scratch_len(k))
        so = 0
    end
    if k >= SQRLO_FULL_THRESHOLD
        sqr!(scratch, so, a, ao, la)
        copyto!(r, ro + 1, scratch, so + 1, k)
        return nothing
    end
    m = mullo_split(k)
    kk = k - m
    lla = min(la, m)
    sqr!(scratch, so, a, ao, lla, scratch, so + 2lla)
    copyto!(r, ro + 1, scratch, so + 1, k)    # 2·min(la, m) > k on this path
    if la > m
        mullo!(scratch, so, a, ao + m, la - m, a, ao, min(la, kk), kk,
               scratch, so + kk + 2)
        add_n!(r, ro + m, r, ro + m, scratch, so, kk)
        add_n!(r, ro + m, r, ro + m, scratch, so, kk)
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
