# Multi-limb division: Knuth Algorithm D quotient/remainder (divrem!) with a
# small-quotient subtraction fast path, dispatching to a divide-and-conquer
# basecase (divrem_dc!, GMP mpn_dcpi1 style) above DC_DIV_THRESHOLD. Built on
# the division kernels (divrem_1!/divrem_2!/divrem_bc!/invert_pi1), the
# shift/add/sub kernels, and mul!/sqr! from mul.jl.

# Quotient/remainder: a (n limbs) ÷ d (m limbs, d[m] ≠ 0), n ≥ m ≥ 1.
# For m ≥ 3, a[n] must be nonzero unless n == m (the small-quotient fast
# path bounds the quotient by a[n]'s bit position).
# Writes n-m+1 quotient limbs (top may be zero) and m remainder limbs
# (unnormalized); a is not modified. scratch holds the shifted numerator
# copy, the shifted divisor for unnormalized d, and the dc block scratch —
# n + 1 + 2m limbs at sco (callers running division in a loop or recursion
# pass their own; the no-scratch method allocates).
function divrem!(q::Memory{Limb}, qo::Int, r::Memory{Limb}, ro::Int,
                 a::Memory{Limb}, ao::Int, n::Int, d::Memory{Limb}, do_::Int, m::Int,
                 scratch::Union{Memory{Limb},Nothing}=nothing, sco::Int=0)
    if m == 1
        @inbounds r[ro+1] = divrem_1!(q, qo, a, ao, n, d[do_+1])
        return nothing
    end
    if m == 2
        r1, r0 = @inbounds divrem_2!(q, qo, a, ao, n, d[do_+2], d[do_+1])
        @inbounds r[ro+1] = r0
        @inbounds r[ro+2] = r1
        return nothing
    end
    # Small-quotient fast path: bitlength(a) - bitlength(d) ≤ 2 bounds the
    # quotient below 8 (a/d < 2^(Δ+1)), so at most 7 subtraction sweeps beat
    # the scratch-alloc + normalize + invert + basecase machinery. Δ ≤ 2 also
    # forces n ≤ m+1 with a[n] < 4, so the value fits r plus one register.
    dbits = magnitude_bits(a, ao, n) - magnitude_bits(d, do_, m)
    if dbits <= 2
        copyto!(r, ro + 1, a, ao + 1, m)
        t = n > m ? (@inbounds a[ao+n]) : zero(Limb)
        c = zero(Limb)
        while t != zero(Limb) || cmp_limbs(r, ro, m, d, do_, m) >= 0
            t -= sub_n!(r, ro, r, ro, d, do_, m)
            c += one(Limb)
        end
        @inbounds q[qo+1] = c
        n > m && (@inbounds q[qo+2] = zero(Limb))
        return nothing
    end
    # Barrett tier (mu_divrem!, defined in invert.jl — included after this file,
    # which Julia resolves at call time). Taken only when we own the scratch:
    # a caller-supplied buffer is sized for the dc path and mu_divrem! needs a
    # larger one, so the one scratch-passing caller (sqrt.jl:105) stays on dc
    # regardless of shape.
    if scratch === nothing && mu_div_worthwhile(n - m + 1, m)
        return mu_divrem!(q, qo, r, ro, a, ao, n, d, do_, m)
    end
    l = leading_zeros(@inbounds d[do_+m])
    nn = n + 1
    if scratch === nothing
        scratch = Memory{Limb}(undef, nn + 2m)
        sco = 0
    end
    if l == 0
        copyto!(scratch, sco + 1, a, ao + 1, n)
        @inbounds scratch[sco+nn] = zero(Limb)
        dv, dvo = d, do_
    else
        @inbounds scratch[sco+nn] = lshift!(scratch, sco, a, ao, n, l)
        lshift!(scratch, sco + nn, d, do_, m, l)
        dv, dvo = scratch, sco + nn
    end
    v = @inbounds invert_pi1(dv[dvo+m], dv[dvo+m-1])
    # qh == 0 either way: Q < β^(nn-m)
    if m >= DC_DIV_THRESHOLD && nn - m >= DC_DIV_PARTIAL_THRESHOLD
        divrem_dc!(q, qo, scratch, sco, nn, dv, dvo, m, v, DC_DIV_THRESHOLD,
                   scratch, sco + nn + m)
    else
        divrem_bc!(q, qo, scratch, sco, nn, dv, dvo, m, v)
    end
    if l == 0
        copyto!(r, ro + 1, scratch, sco + 1, m)
    else
        rshift!(r, ro, scratch, sco, m, l)
    end
    return nothing
end

# Schoolbook → divide-and-conquer division crossover, in limbs (GMP's
# DC_DIV_QR_THRESHOLD analogue; tuned by bench/bench_dc_thr.jl). It gates the
# *divisor* side only: the quotient side uses the lower
# DC_DIV_PARTIAL_THRESHOLD, because a short quotient over a long divisor does
# NOT cost the same either way. Both arms are O(qn·m) in operation count, but
# schoolbook spends it in row kernels while the leading-partial arm spends it
# in mul!(qn, m-qn), which is subquadratic. Requiring qn >= 100 here used to
# leave a cliff where a longer quotient divided faster (m = 1024: 27.2us at
# qn = 96 against 14.7us at qn = 112).
# Balanced 2m/m sweep: schoolbook wins through m = 96, ties at m = 128, and
# falls behind from m = 192 (1.1-3x GMP by m = 2048 vs dc's 0.8-1.0x); as the
# recursion cutoff, 100-110 also edges out lower values at large m.
const DC_DIV_THRESHOLD = 100

# Balanced 2n/n divide-and-conquer division step (GMP mpn_dcpi1_div_qr_n).
# u[uo+1..uo+2n] ÷ d[do_+1..do_+n], d normalized, v = invert_pi1 of its top
# two limbs. Writes q[qo+1..qo+n], leaves the n-limb remainder in
# u[uo+1..uo+n], returns the extra top quotient bit qh. scratch holds the
# n-limb cross products at so; recursion levels share it (a level touches it
# only after its child returns). thr >= 4 keeps the divrem_bc! basecase at
# m >= 2.
#
# Split n = hi + lo. Dividing the top 2·hi limbs by the top hi limbs of d
# overshoots the true high quotient block by at most 2 (the ignored d_lo only
# makes the divisor larger): subtract the cross product q_hi·d_lo — plus
# qh·d_lo at β^hi for the implicit qh row — and repair each resulting borrow
# by adding d back and decrementing the block. The same step on the remaining
# n + lo live limbs yields the low block. The pi1 inverse stays valid down
# the recursion because every divisor suffix keeps d's top two limbs.
function divrem_dc_n!(q::Memory{Limb}, qo::Int, u::Memory{Limb}, uo::Int,
                      d::Memory{Limb}, do_::Int, n::Int, v::Limb,
                      scratch::Memory{Limb}, so::Int, thr::Int)
    lo = n >> 1
    hi = n - lo
    qh = hi < thr ? divrem_bc!(q, qo + lo, u, uo + 2lo, 2hi, d, do_ + lo, hi, v) :
                    divrem_dc_n!(q, qo + lo, u, uo + 2lo, d, do_ + lo, hi, v, scratch, so, thr)
    mul!(scratch, so, q, qo + lo, hi, d, do_, lo)          # q_hi × d_lo, n limbs
    cy = sub_n!(u, uo + lo, u, uo + lo, scratch, so, n)
    if qh != zero(Limb)
        cy += sub_n!(u, uo + n, u, uo + n, d, do_, lo)
    end
    while cy != zero(Limb)
        qh -= sub_1!(q, qo + lo, q, qo + lo, hi, one(Limb))
        cy -= add_n!(u, uo + lo, u, uo + lo, d, do_, n)
    end
    ql = lo < thr ? divrem_bc!(q, qo, u, uo + hi, 2lo, d, do_ + hi, lo, v) :
                    divrem_dc_n!(q, qo, u, uo + hi, d, do_ + hi, lo, v, scratch, so, thr)
    mul!(scratch, so, d, do_, hi, q, qo, lo)               # q_lo × d_lo', hi >= lo
    cy = sub_n!(u, uo, u, uo, scratch, so, n)
    if ql != zero(Limb)
        cy += sub_n!(u, uo + lo, u, uo + lo, d, do_, hi)
    end
    while cy != zero(Limb)
        sub_1!(q, qo, q, qo, lo, one(Limb))   # borrow folds into ql's correction
        cy -= add_n!(u, uo, u, uo, d, do_, n)
    end
    return qh
end

# Schoolbook → cross-product crossover for the *leading partial* block, which
# is a different question from the balanced DC_DIV_THRESHOLD and needs its own
# constant. The schoolbook arm costs O(s·m) with row-kernel constants; the
# cross-product arm costs a cheap 2s/s division plus mul!(s, m-s), so it wins
# as soon as mul! is subquadratic — a far lower bar than the balanced case,
# and near-independent of m.
#
# Gating this on DC_DIV_THRESHOLD instead left a cliff where a *longer*
# quotient divided faster. divrem! ns by quotient limbs, before → after:
#
#   qn        32      48      64      80      96      112
#   m=1024  9288    14127   18635   24497   26611   14568
#      ->   9288    12213   12224   12935   13505   14708
#   m=2048 16611    28544   37362   42591   53662   26059
#      ->  16982    23826   23575   24247   24928   26110
#
# 48 rather than 32: at 32 the cross-product arm is still behind (m = 2048 went
# 16.6us -> 21.1us), and the crossover measures out at ~44.
# Must stay >= 4 so divrem_dc_n!'s halves keep divrem_bc! at m >= 2.
const DC_DIV_PARTIAL_THRESHOLD = 48

# Leading partial quotient block: u[uo+1..uo+m+s] ÷ d (full m limbs), s <= m.
# Writes q[qo+1..qo+s], leaves the m-limb remainder in u[uo+1..uo+m], returns
# qh. s == m is the balanced step; very small s stays schoolbook (O(s·m),
# one-off); otherwise divide the top 2s limbs by the top s limbs of d, then
# subtract the cross product q·d_lo with the same add-back repair (GMP
# mpn_dcpi1_div_qr's qn < dn arm).
function divrem_dc_partial!(q::Memory{Limb}, qo::Int, u::Memory{Limb}, uo::Int,
                            d::Memory{Limb}, do_::Int, m::Int, s::Int, v::Limb,
                            scratch::Memory{Limb}, so::Int, thr::Int)
    s == m && return divrem_dc_n!(q, qo, u, uo, d, do_, m, v, scratch, so, thr)
    s < min(thr, DC_DIV_PARTIAL_THRESHOLD) &&
        return divrem_bc!(q, qo, u, uo, m + s, d, do_, m, v)
    qh = divrem_dc_n!(q, qo, u, uo + (m - s), d, do_ + (m - s), s, v, scratch, so, thr)
    if s >= m - s                                          # q × d_lo, m limbs
        mul!(scratch, so, q, qo, s, d, do_, m - s)
    else
        mul!(scratch, so, d, do_, m - s, q, qo, s)
    end
    cy = sub_n!(u, uo, u, uo, scratch, so, m)
    if qh != zero(Limb)
        cy += sub_n!(u, uo + s, u, uo + s, d, do_, m - s)
    end
    while cy != zero(Limb)
        qh -= sub_1!(q, qo, q, qo, s, one(Limb))
        cy -= add_n!(u, uo, u, uo, d, do_, m)
    end
    return qh
end

# Divide-and-conquer quotient/remainder with divrem_bc!'s exact contract:
# u (nn limbs, destroyed; remainder left in u[uo+1..uo+m]) ÷ normalized
# m-limb d with v = invert_pi1 of its top two limbs; writes q[1..nn-m],
# returns the extra top quotient bit qh. Requires nn > m. The quotient is
# peeled from the top: a leading partial block of s limbs (qn reduced mod m
# into [1, m]), then full balanced 2m/m blocks — each later window tops out
# with the previous remainder (< d), so only the leading block can set qh.
function divrem_dc!(q::Memory{Limb}, qo::Int, u::Memory{Limb}, uo::Int, nn::Int,
                    d::Memory{Limb}, do_::Int, m::Int, v::Limb,
                    thr::Int=DC_DIV_THRESHOLD,
                    scratch::Memory{Limb}=Memory{Limb}(undef, m), so::Int=0)
    qn = nn - m
    thr = max(thr, 4)
    s = qn <= m ? qn : qn - m * ((qn - 1) ÷ m)
    off = qn - s
    qh = divrem_dc_partial!(q, qo + off, u, uo + off, d, do_, m, s, v, scratch, so, thr)
    while off > 0
        off -= m
        divrem_dc_n!(q, qo + off, u, uo + off, d, do_, m, v, scratch, so, thr)
    end
    return qh
end

# Schoolbook → divide-and-conquer crossover for the *approximate* engines. It is
# far higher than DC_DIV_THRESHOLD because divappr_bc! truncates every row's
# submul to the triangle (kernels/div.jl:345), a constant-factor saving that
# divappr_dc! largely forfeits — all its blocks above the bottom-most run the
# exact divrem_dc_partial!, remainder work included.
#
# divappr_bc!/divappr_dc! on balanced nn = 2m+1, by divisor limbs m:
#
#   m      100   128   192   256   384   512   640   768   896
#   ratio  0.87  0.81  0.90  0.91  1.02  1.09  1.31  1.47  1.50
#
# Set from a paired isqrt A/B (identical seeded inputs, this constant the only
# variable), by the top-level divisor width hh it reaches. hh = 64 and hh = 448
# take the same engine either way and serve as controls, putting the noise floor
# at 0.4%:
#
#   hh        64*   128    192    256    352    448*
#   thr=100  2.72  7.47   13.40  24.22  38.02  51.79   us
#   thr=384  2.71  6.75   12.81  22.34  38.80  51.63
#
# So schoolbook is worth 4.6-10.7% through hh = 256 but loses 2% by hh = 352 --
# 320 takes the wins and leaves 352 on the recursion. Sharing DC_DIV_THRESHOLD
# here (as sqrt did before this constant existed) gave up that whole band.
const DIVAPPR_DC_THRESHOLD = 320

# Approximate leading quotient block, no remainder: writes s quotient limbs q̂
# for the top m+s live limbs of u by the m-limb normalized d, one-sided with
# q_true ≤ q̂ ≤ q_true + E; u above uo is destroyed and holds nothing
# meaningful. Returns the extra top quotient bit/carry (the over-approximation
# of a maximal true quotient can carry out; callers fold it).
#
# E ≤ ~20 for any feasible size: entry and per-level divisor truncation
# contribute ≤ 1 each (numerator ≤ β^qn·d against a kept top ≥ β^(t-1),
# t = qn+2), the triangle basecase ≤ 2, and the dc recursion halves the block
# per level. sqrt_root_cert allows 32, so it has slack.
#
# Structure: truncate the divisor to its top s+2 limbs — only those can move
# the quotient by more than 1 ulp, since the dropped tail is < β^(m-s-2) against
# a divisor ≥ β^(m-1) — then peel the top ⌈s/2⌉ quotient limbs exactly with
# divrem_dc_partial! (their remainder feeds the rest; approximating them would
# scale the error by β^s2), and recurse on the bottom half — the recursion is
# where the remainder work is saved.
function divappr_dc_partial!(q::Memory{Limb}, qo::Int, u::Memory{Limb}, uo::Int,
                             d::Memory{Limb}, do_::Int, m::Int, s::Int, v::Limb,
                             scratch::Memory{Limb}, so::Int, thr::Int)
    if m > s + 2
        drop = m - (s + 2)
        return divappr_dc_partial!(q, qo, u, uo + drop, d, do_ + drop, s + 2, s,
                                   v, scratch, so, thr)
    end
    s < thr && return divappr_bc!(q, qo, u, uo, m + s, d, do_, m, v)
    s2 = s >> 1
    qh = divrem_dc_partial!(q, qo + s2, u, uo + s2, d, do_, m, s - s2, v,
                            scratch, so, thr)
    c = divappr_dc_partial!(q, qo, u, uo, d, do_, m, s2, v, scratch, so, thr)
    if c != zero(Limb)   # rare: the lo block's over-approximation carried out
        qh += add_1!(q, qo + s2, q, qo + s2, s - s2, c)
    end
    return qh
end

# Approximate quotient with divrem_dc!'s peeling and contract, minus the
# remainder: all quotient blocks above the bottom-most are computed exactly
# (their remainders feed lower blocks), only the bottom block runs the
# approximate recursion. u is destroyed, holds no remainder.
function divappr_dc!(q::Memory{Limb}, qo::Int, u::Memory{Limb}, uo::Int, nn::Int,
                     d::Memory{Limb}, do_::Int, m::Int, v::Limb,
                     thr::Int=DC_DIV_THRESHOLD,
                     scratch::Memory{Limb}=Memory{Limb}(undef, m), so::Int=0)
    qn = nn - m
    thr = max(thr, 4)
    qn <= m && return divappr_dc_partial!(q, qo, u, uo, d, do_, m, qn, v, scratch, so, thr)
    s = qn - m * ((qn - 1) ÷ m)
    off = qn - s
    qh = divrem_dc_partial!(q, qo + off, u, uo + off, d, do_, m, s, v, scratch, so, thr)
    while off > m
        off -= m
        divrem_dc_n!(q, qo + off, u, uo + off, d, do_, m, v, scratch, so, thr)
    end
    c = divappr_dc_partial!(q, qo, u, uo, d, do_, m, m, v, scratch, so, thr)
    if c != zero(Limb)
        qh += add_1!(q, qo + m, q, qo + m, qn - m, c)
    end
    return qh
end

# These engines have no divrem!-style entry wrapper: their only consumer is
# sqrt.jl, which drives them directly off its own normalized numerator buffer,
# supplying the appended zero limb that keeps the over-approximated quotient
# inside nn-m limbs. test_algorithms.jl carries an equivalent driver so they
# keep direct coverage over arbitrary shapes.
