# Multi-limb approximate reciprocal (GMP mpn_invertappr): the Newton-doubling
# generalization of the invert_limb/invert_pi1 kernels, and the object a
# Barrett/Newton division tier is built on. Built on mul! from mul.jl and the
# add/sub kernels.
#
# Design notes and the derivation live in
# docs/superpowers/specs/2026-08-22-invertappr-design.md.

# Newton recursion → direct-division basecase crossover, in limbs (GMP's
# INV_NEWTON_THRESHOLD analogue; tuned by bench/bench_kernels.jl invert).
# Below it, one 2n/n division beats the recursion outright. Newton's ratio vs
# the basecase: 0.96 at 80, 0.98 at 88, 0.91 at 96, 0.93 at 104, 1.03 at 112,
# 1.00 at 120, 0.98 at 128, 0.92 at 160, 0.85 at 224, 0.70 at 512, 0.66 at
# 1024; 1.07 at 64 is the last clear loss, which is what sets the value.
#
# The band above 72 is not monotone — 112 and 120 sit at 1.03/1.00 where both
# sides cross Karatsuba/NTT boundaries — so this trades one ~3% dip for 4-9%
# across the rest of 80-136. A more conservative 128 would forgo that whole
# pocket. Before E was windowed in the assembly step this sat at 232.
const INV_NEWTON_THRESHOLD = 72

# Scratch limbs the basecase needs at sco: just the 2n-limb numerator it
# divides in place.
invertappr_bc_scratch_len(n::Int) = n <= 1 ? 0 : 2n

# Scratch limbs invertappr! needs at sco. The recursive call completes before
# either working region is touched, so every level shares one buffer: S (the
# n+h-limb product/residual) followed by W (the n+4-limb assembly product).
# A child that bottoms out in the basecase can need 8h+2 limbs, which exceeds
# the parent's own need when h ≈ n/2 — hence the max, not just the local size.
# Must clamp thr exactly as invertappr! does or the sizes disagree.
function invertappr_scratch_len(n::Int, thr::Int=INV_NEWTON_THRESHOLD)
    thr = max(thr, 4)
    n <= thr && return invertappr_bc_scratch_len(n)
    h = (n >> 1) + 1
    return max(2n + h + 4, invertappr_scratch_len(h, thr))
end

# Basecase: X = ⌊(β^2n - 1)/A⌋ - β^n by one exact 2n/n division — no error at
# all. Exact division rather than the divappr engines, whose ~20 ulps would swamp
# the ≤ 1 contract.
#
# Goes straight to divrem_bc! rather than through divrem!. d is already
# normalized by contract, so divrem! would only add work we can skip: it copies
# the numerator into its own scratch (we generate ours in place, and the kernel
# destroys it anyway), and it materializes an n-limb remainder we discard. The
# quotient is in [β^n, 2β^n - 1], so the n limbs it writes are exactly X and the
# returned qh is the (discarded) leading 1 — no n+1-limb quotient buffer and no
# copy out.
#
# Schoolbook is the only arm needed: invertappr! enters here only at
# n <= INV_NEWTON_THRESHOLD (72), below DC_DIV_THRESHOLD (100), so a dc arm was
# unreachable outside tests that force thr = n. Keeping it would tie invert.jl
# to the dc engines for no production benefit.
function invertappr_bc!(ip::Memory{Limb}, io::Int, d::Memory{Limb}, do_::Int, n::Int,
                        scratch::Memory{Limb}, sco::Int)
    if n == 1
        @inbounds ip[io+1] = invert_limb(d[do_+1])
        return nothing
    end
    @inbounds for i in 1:2n
        scratch[sco+i] = typemax(Limb)          # numerator β^2n - 1, destroyed
    end
    v = @inbounds invert_pi1(d[do_+n], d[do_+n-1])
    divrem_bc!(ip, io, scratch, sco, 2n, d, do_, n, v)
    return nothing
end

# Approximate reciprocal of the normalized m-limb A = d[do_+1..do_+n]
# (d[do_+n] top bit set): writes n limbs X to ip[io+1..io+n] with
#
#     0 ≤ V - X ≤ 1,   V = ⌊(β^2n - 1)/A⌋ - β^n
#
# The bound is 1, matching GMP's mpn_invertappr guarantee; measured over 6143
# cases (n = 1..400 plus 511/512/513/1000, random plus the normalized
# endpoints) and asserted by the test.
#
# The -1 in V matters: without it A = β^n/2 gives V = β^n, which does not fit
# in n limbs — it is the same convention invert_limb uses, and n == 1 reduces
# to that value (via the basecase, which is exact). ip must not alias d or
# scratch;
# d is read-only. scratch needs invertappr_scratch_len(n, thr) limbs at sco
# (allocated when not passed). thr is the basecase crossover, exposed only so
# tests can drive the recursion at small n (cf. divrem_dc!'s thr); clamped to
# >= 4, so the recursion body only ever runs at n >= 5. That is what lets the
# split be a bare (n>>1)+1 (it is already < n by then) and the assembly window
# below assume h >= 3.
#
# Newton doubling. With Z = β^n + X and Z_h = β^h + X_h the recursive inverse
# of A's top h limbs, Z_h ≈ β^(h+n)/A, so z₀ = Z_h/β^(h+n) approximates 1/A;
# one step z₁ = z₀(2 - A z₀) scaled by β^2n gives
#
#     Z = Z_h·β^l + Z_h·E/β^2h,   E = β^(h+n) - A·Z_h
#
# E is ~β^n (z₀ is accurate to about h limbs), so the correction is the l limbs
# sitting below the Z_h·β^l term. E is formed as a complement of T = A·Z_h
# rather than directly, so the arithmetic stays unsigned: T ≥ β^(n+h) — the
# case where E would be negative — shows up as a carry out of T's top limb.
function invertappr!(ip::Memory{Limb}, io::Int, d::Memory{Limb}, do_::Int, n::Int,
                     scratch::Union{Memory{Limb},Nothing}=nothing, sco::Int=0,
                     thr::Int=INV_NEWTON_THRESHOLD)
    if scratch === nothing
        scratch = Memory{Limb}(undef, invertappr_scratch_len(n, thr))
        sco = 0
    end
    thr = max(thr, 4)
    if n <= thr
        return invertappr_bc!(ip, io, d, do_, n, scratch, sco)
    end
    # X_h lands directly in the top h limbs of the output, so the Z_h·β^l term
    # of the assembly below needs no move.
    l = n - invertappr_seed_len(n)
    invertappr!(ip, io + l, d, do_ + l, invertappr_seed_len(n), scratch, sco, thr)
    return invertappr_step!(ip, io, d, do_, n, scratch, sco)
end

# Seed width one Newton step at size n consumes: invertappr_step! requires
# ip[io+l+1 .. io+n] to already hold the reciprocal of d's top
# invertappr_seed_len(n) limbs. The +1 is the guard limb — see the split note
# above; without it the error occasionally blows up to 74-108 ulps.
invertappr_seed_len(n::Int) = (n >> 1) + 1

# Scratch limbs one step needs at sco: S/E (n + seed) followed by W (n + 4).
# NOT interchangeable with invertappr_scratch_len(n), which returns the
# *basecase* size for n <= thr — a basecase never runs a step, and at n = 5
# that size is 15 limbs against the 17 a step writes.
invertappr_step_scratch_len(n::Int) = 2n + invertappr_seed_len(n) + 4

# One Newton doubling step, factored out of invertappr! so a caller that
# already holds a partial reciprocal can refine it without rebuilding from
# scratch. On entry ip[io+l+1 .. io+n] (l = n - h) holds X_h, the reciprocal of
# d's top h limbs; on return ip[io+1 .. io+n] holds the n-limb reciprocal of the
# full d. Requires 3 <= h <= n-1 and invertappr_step_scratch_len(n) limbs of
# scratch at sco (which is sized for the default h; a larger h needs less).
#
# h is the seed width and must be at least invertappr_seed_len(n) — half plus
# the guard limb. The guard is not slack: a ladder of steps seeded at exactly
# half compounds, measured at 2, 3, 10, 97, 209 and 2.6e15 ulps for chains of
# 64..2048 limbs. A single step from an accurate seed is fine at exactly half,
# which makes this easy to mis-measure; the ladder is what matters. A seed
# *wider* than the default is always safe — it is strictly more information —
# and sqrt's ladder uses that to widen by one limb before doubling.
#
# X_h is consumed in place, which is what makes the Z_h·β^l term free.
function invertappr_step!(ip::Memory{Limb}, io::Int, d::Memory{Limb}, do_::Int,
                          n::Int, scratch::Memory{Limb}, sco::Int,
                          h::Int=invertappr_seed_len(n))
    # h in [3, n-1] and l >= 1; the mul operand order below adapts to whichever
    # of ke = l+3 and h is longer (with the default h, ke always leads).
    l = n - h
    so = sco                # S/E: n+h limbs
    wo = sco + n + h        # W:   n+4 limbs

    # T = A·Z_h = A·β^h + A·X_h, with the overflow bit carried in cy.
    mul!(scratch, so, d, do_, n, ip, io + l, h)
    cy = add_n!(scratch, so + h, scratch, so + h, d, do_, n)
    # cy != 0 ⟺ E < 0: Z_h overshot β^(h+n)/A. T overshoots by ~β^n and A is
    # also ~β^n, so this converges in a handful of steps (≤ 4 measured).
    while cy != zero(Limb)
        sub_1!(ip, io + l, ip, io + l, h, one(Limb))          # Z_h -= 1
        b = sub_n!(scratch, so, scratch, so, d, do_, n)       # T  -= A
        if b != zero(Limb)
            b = sub_1!(scratch, so + n, scratch, so + n, h, one(Limb))
        end
        cy -= b
    end
    # E = β^(n+h) - T, but only E's window [h-1 .. n+1] is ever read below, so
    # only that range is complemented. The borrow out of the discarded low h-2
    # limbs is 1 iff any of them is nonzero — a read-only OR reduction instead
    # of a read-modify-write pass over all n+h limbs. Exact, not approximate:
    # ⌊E/β^(h-2)⌋ = β^(n+2) - ⌊T/β^(h-2)⌋ - b, and mod β^ke the β^(n+2) term
    # vanishes (ke <= n), leaving exactly this negation-with-borrow; E < β^(n+1)
    # so the window loses nothing off the top.
    acc = zero(Limb)
    @inbounds for i in 1:(h-2)
        acc |= scratch[so+i]
    end
    brw = ifelse(acc == zero(Limb), zero(Limb), one(Limb))
    @inbounds for i in (h-1):(n+1)
        scratch[so+i], brw = sub_limb_b(zero(Limb), scratch[so+i], brw)
    end

    # The assembly product runs on a window of E, not all n+h limbs of it.
    #
    # E < 2A ≤ 2(β^n - 1), so E is at most n+1 limbs with a top limb of 0 or 1.
    # Proof, with Q = ⌊(β^2h - 1)/A_h⌋ so the child's contract gives Z_h ≥ Q-1:
    # if the correction loop above ran, it exited the first time A·Z ≤ β^(n+h),
    # so the previous iterate had A·(Z+1) > β^(n+h) and E < A. If it never ran,
    # Z = Z_h ≥ Q-1; the division remainder is ≤ A_h - 1 so A_h·Q ≥ β^2h - A_h,
    # hence A_h(Q-1) ≥ β^2h - 2A_h, and with A ≥ A_h·β^l and 2h+l = h+n,
    # A·Z ≥ β^(h+n) - 2A_h·β^l, i.e. E ≤ 2A_h·β^l ≤ 2A.
    #
    # Only E's top l+1 limbs survive the β^2h shift — the rest cannot move the
    # answer (measured, and the neglected tail is worth < 2β^-2 ulp). Dropping
    # the low h-2 limbs and keeping through limb n+1 covers the whole bound
    # above with no runtime width check, and is a 3x smaller product than the
    # full Z_h·E.
    eo = so + h - 2         # drop E's low h-2 limbs
    ke = l + 3              # reaches limb n+1
    s = h + 2               # = 2h - (h-2): the shift left after truncation
    # W = Z_h·Ehi = X_h·Ehi + β^h·Ehi. The product is h+ke = n+3 limbs, and
    # W < 4β^(n+2) < β^(n+3) so the add cannot carry out — limb n+4 catches it
    # only to keep the slice below a fixed width.
    if ke >= h
        mul!(scratch, wo, scratch, eo, ke, ip, io + l, h)
    else
        mul!(scratch, wo, ip, io + l, h, scratch, eo, ke)
    end
    @inbounds scratch[wo+n+4] =
        add_n!(scratch, wo + h, scratch, wo + h, scratch, eo, ke)
    # C = ⌊W/β^s⌋, l+2 limbs (C < 4β^l): l below X_h, the rest carrying into it.
    copyto!(ip, io + 1, scratch, wo + s + 1, l)
    add_into!(ip, io + l, h, scratch, wo + s + l, 2)
    return nothing
end

# ---- Barrett (mu) division ------------------------------------------------
# Newton/Barrett division on top of invertappr!: GMP's mpn_mu_div_qr. Where
# divrem_dc! spends a recursive division per m-limb quotient block, this spends
# two multiplications, having paid for one reciprocal up front. It is therefore
# an O(M(n)) algorithm against divrem_dc!'s O(M(n)·log n), which is why the
# crossover exists at all and why it keeps widening: measured D/M is 4.3 at 512,
# 6.1 at 2048, 7.8 at 8192.

# Scratch limbs mu_div_step! needs at so: the m+s+2-limb quotient-estimate
# product followed by the m+s-limb q̂·d product. s <= m.
mu_div_step_scratch_len(m::Int) = 4m + 2

# Barrett-vs-divrem! crossover. The reciprocal is paid once and amortizes over
# k = ceil(qn/m) blocks, so the crossover moves with the *shape*, not just the
# divisor size: mu wins once inv(m) < k·(D(m) - 2·M(m)). Measured mu/divrem!
# ratios (bench/bench_kernels.jl mudiv), by divisor limbs m:
#
#   shape         256   384   512   768   1024
#   2m/m  (k=1)   1.50  1.28  1.09  1.01  0.91
#   3m/m  (k=2)   1.08  0.90  0.79  0.75  0.66
#   5m/m  (k=4)   0.87  0.73  0.63  0.62  0.53
#
# Each clause below is anchored on a measured crossover in one of those rows.
# k=3 is not measured, so it is served by the k=2 clause (conservative).
#
const MU_DIV_THRESHOLD = 1024       # qn >= m, single block
const MU_DIV_2BLK_THRESHOLD = 384   # needs qn >= 2m
const MU_DIV_4BLK_THRESHOLD = 256   # needs qn >= 4m

# Short quotients (qn < m) are a separate regime: one block, and mu_inv_size
# trims the reciprocal to qn+2 limbs, so the deciding sub-problem is size qn,
# not m. Both algorithms then pay the same q̂·d cross product, and mu's estimate
# (inv(qn) + M(qn)) beats dc's (a 2qn/qn division, D(qn)) only once Newton's
# O(M) has overtaken dc's O(M·log). That crossover is in qn and is nearly
# independent of m — mu/dc measured at qn / m:
#
#   qn        256   384   512   640   768   1024  1536  2048
#   m=1024    1.16  1.05  1.00  0.97  0.93  0.90    -     -
#   m=2048    1.21  1.09  1.01  0.95  0.92  0.85  0.81  0.77
#   m=4096    1.21  1.12  1.05  1.02  0.98  0.88  0.78  0.74
#
# 768 is the first row that wins at every m (640 still loses at m = 4096). The
# margin shrinks as m grows with qn fixed, since the shared cross product comes
# to dominate; the real wins are at qn >= 1024.
const MU_DIV_QN_THRESHOLD = 768

mu_div_worthwhile(qn::Int, m::Int) =
    qn < m ? qn >= MU_DIV_QN_THRESHOLD :
             (m >= MU_DIV_THRESHOLD ||
              (m >= MU_DIV_2BLK_THRESHOLD && qn >= 2m) ||
              (m >= MU_DIV_4BLK_THRESHOLD && qn >= 4m))

# One Barrett block: divide the m+s-limb window u[uo+1..uo+m+s] by the
# normalized m-limb d, given iv, the m-limb approximate reciprocal of d
# (so Z = β^m + iv ≈ β^2m/d). Requires U < d·β^s, i.e. the quotient fits in s
# limbs, and 1 <= s <= m. Writes q[qo+1..qo+s] and leaves the m-limb remainder
# in u[uo+1..uo+m]; the window above that is left zero.
#
#     U_hi = ⌊U/β^(m-1)⌋   (s+1 limbs)
#     q̂    = ⌊U_hi·Z/β^(m+1)⌋
#
# q̂ never overshoots: Z ≤ ⌊(β^2m - 1)/d⌋ < β^2m/d, so q̂ ≤ U/d and the fixups
# below are add-only. Undershoot is at most 5 — one from truncating U to U_hi,
# up to three from Z's own error (invertappr!'s 1 ulp plus the two floors relating
# it to β^2m/d), and one more from U/β^2m < 1 — so the loop is O(1) passes.
#
# Split into the estimate and the fixup so mu_divappr_core! can reuse the
# estimate alone for its bottom block, where no remainder is wanted.
@inline function mu_div_est!(q::Memory{Limb}, qo::Int, u::Memory{Limb}, uo::Int, s::Int,
                             m::Int, iv::Memory{Limb}, ivo::Int, ii::Int,
                             scratch::Memory{Limb}, po::Int)
    uh = uo + m - 1                    # U_hi lives at u[uh+1 .. uh+s+1]
    # P = U_hi·Z = U_hi·iv + U_hi·β^ii. Z is never materialized: the β^ii term
    # is the same add-at-offset trick invertappr! uses internally.
    if ii >= s + 1
        mul!(scratch, po, iv, ivo, ii, u, uh, s + 1)
    else
        mul!(scratch, po, u, uh, s + 1, iv, ivo, ii)
    end
    @inbounds scratch[po+s+ii+2] =
        add_n!(scratch, po + ii, scratch, po + ii, u, uh, s + 1)
    # q̂ = ⌊P/β^(ii+1)⌋. With ii == m it is ≤ ⌊U/d⌋ and fits in s limbs; with a
    # truncated divisor it can reach β^s, which is clamped to β^s - 1 (still
    # ≥ the true quotient, so the caller's add-back still fixes it).
    est = po + ii + 1
    @inbounds if ii < m && scratch[est+s+1] != zero(Limb)
        for i in 1:s
            q[qo+i] = typemax(Limb)
        end
    else
        copyto!(q, qo + 1, scratch, est + 1, s)
    end
    return nothing
end

function mu_div_step!(q::Memory{Limb}, qo::Int, u::Memory{Limb}, uo::Int, s::Int,
                      d::Memory{Limb}, do_::Int, m::Int,
                      iv::Memory{Limb}, ivo::Int, ii::Int,
                      scratch::Memory{Limb}, so::Int)
    po = so                            # P: s+ii+2 limbs
    wo = so + s + ii + 2               # W: m+s limbs
    mu_div_est!(q, qo, u, uo, s, m, iv, ivo, ii, scratch, po)
    # R = U - q̂·d, against the *full* divisor. R < 6d < β^(m+1), so the
    # window's limbs above m+1 end up zero either way.
    if m >= s
        mul!(scratch, wo, d, do_, m, q, qo, s)
    else
        mul!(scratch, wo, q, qo, s, d, do_, m)
    end
    cy = sub_n!(u, uo, u, uo, scratch, wo, m + s)
    # Add-back arm. Unreachable when ii == m (q̂ ≤ ⌊U/d⌋ there), but a truncated
    # divisor only ever makes the estimate too large, by at most 1 with the two
    # guard limbs mu_inv_size leaves.
    @inbounds while cy != zero(Limb)
        sub_1!(q, qo, q, qo, s, one(Limb))
        c = add_n!(u, uo, u, uo, d, do_, m)
        c != zero(Limb) && (c = add_1!(u, uo + m, u, uo + m, s, c))
        cy -= c
    end
    @inbounds while u[uo+m+1] != zero(Limb) || cmp_limbs(u, uo, m, d, do_, m) >= 0
        b = sub_n!(u, uo, u, uo, d, do_, m)
        u[uo+m+1] -= b
        add_1!(q, qo, q, qo, s, one(Limb))
    end
    return nothing
end

# Blocked Barrett division with divrem_dc!'s contract: u (nn limbs, destroyed;
# m-limb remainder left in u[uo+1..uo+m]) ÷ normalized m-limb d, with iv the
# m-limb approximate reciprocal; writes q[1..nn-m]. Requires nn > m and
# Q < β^(nn-m) — the caller's appended limb guarantees it, so there is no qh.
#
# The quotient is peeled from the top exactly as divrem_dc! peels it: a leading
# partial block of s limbs (qn reduced mod m into [1, m]), then full m-limb
# blocks. That is what supplies mu_div_step!'s U < d·β^s precondition — the
# leading window inherits it from Q < β^qn, and every later window tops out
# with the previous block's remainder, which is < d.
function mu_divrem_core!(q::Memory{Limb}, qo::Int, u::Memory{Limb}, uo::Int, nn::Int,
                         d::Memory{Limb}, do_::Int, m::Int,
                         iv::Memory{Limb}, ivo::Int, ii::Int,
                         scratch::Memory{Limb}, so::Int)
    qn = nn - m
    if ii < m
        # short quotient: one block against a reciprocal of only d's top ii
        # limbs (GMP's mpn_mu_div_qr_choose_in). Blocking would need ii ≥ s+2
        # per block, which a single block of s = qn already satisfies.
        return mu_div_step!(q, qo, u, uo, qn, d, do_, m, iv, ivo, ii, scratch, so)
    end
    s = qn <= m ? qn : qn - m * ((qn - 1) ÷ m)
    off = qn - s
    mu_div_step!(q, qo + off, u, uo + off, s, d, do_, m, iv, ivo, ii, scratch, so)
    while off > 0
        off -= m
        mu_div_step!(q, qo + off, u, uo + off, m, d, do_, m, iv, ivo, ii, scratch, so)
    end
    return nothing
end

# Barrett approximate quotient, with divappr_dc!'s contract: u (nn limbs,
# destroyed, holds no remainder afterwards) ÷ normalized m-limb d, writing
# qn = nn-m limbs to q with
#
#     q_true ≤ q̂ ≤ q_true + 6
#
# Requires nn > m and Q < β^qn (the caller's appended zero limb).
#
# Every block above the bottom-most runs the full mu_div_step! — their
# remainders feed the blocks below — and only the bottom one stops at the
# estimate, skipping its q̂·d product and correction entirely. That is the whole
# saving over mu_divrem!: one of the two multiplications per block, for one
# block.
#
# The estimate undershoots by at most 5, so adding 5 turns it into the
# one-sided *over*-approximation divappr's callers require (sqrt.jl's guard-limb
# certificate depends on the sign).
#
# iv is the full m limbs and is supplied, never built here: sqrt's ladder
# already holds inv of exactly this divisor (the child's root), so its only
# caller pays nothing for it. Using the full width also keeps the estimate free
# of the truncated-divisor overshoot that would complicate the bound above.
# scratch needs mu_div_step_scratch_len(m) limbs at wso.
function mu_divappr_core!(q::Memory{Limb}, qo::Int, u::Memory{Limb}, uo::Int, nn::Int,
                          d::Memory{Limb}, do_::Int, m::Int,
                          iv::Memory{Limb}, ivo::Int,
                          scratch::Memory{Limb}, wso::Int)
    qn = nn - m
    s = qn <= m ? qn : qn - m * ((qn - 1) ÷ m)
    off = qn - s
    if off > 0
        mu_div_step!(q, qo + off, u, uo + off, s, d, do_, m, iv, ivo, m, scratch, wso)
        while off > m
            off -= m
            mu_div_step!(q, qo + off, u, uo + off, m, d, do_, m, iv, ivo, m, scratch, wso)
        end
        off -= m
        s = m
    end
    # bottom block: estimate only, then lift to a one-sided over-approximation
    mu_div_est!(q, qo, u, uo, s, m, iv, ivo, m, scratch, wso)
    add_1!(q, qo, q, qo, qn, Limb(5))
    return nothing
end

# Reciprocal size: the full divisor when the quotient is at least as long (the
# blocked case, where a shorter reciprocal would only force more blocks and so
# more q̂·d products against the full divisor), otherwise just enough to pin the
# quotient — qn limbs plus two guard limbs, which bounds the truncated
# divisor's overshoot at 1 (the relative error is < 4·β^(qn-ii)).
mu_inv_size(qn::Int, m::Int) = qn >= m ? m : min(m, qn + 2)

# Scratch limbs mu_divrem! needs at sco: the n+1-limb shifted numerator, the
# m-limb shifted divisor, the m-limb reciprocal, then whichever of the
# reciprocal's own scratch and the step scratch is larger (they are used in
# sequence, never at once).
mu_divrem_scratch_len(n::Int, m::Int) =
    (n + 1) + 2m + max(invertappr_scratch_len(m), mu_div_step_scratch_len(m))

# Barrett quotient/remainder with divrem!'s operand contract: a (n limbs) ÷ d
# (m limbs, d[m] ≠ 0), n > m ≥ 2. Writes n-m+1 quotient limbs (top may be zero)
# and m remainder limbs; a is not modified. scratch needs
# mu_divrem_scratch_len(n, m) limbs at sco (allocated when not passed).
function mu_divrem!(q::Memory{Limb}, qo::Int, r::Memory{Limb}, ro::Int,
                    a::Memory{Limb}, ao::Int, n::Int, d::Memory{Limb}, do_::Int, m::Int,
                    scratch::Union{Memory{Limb},Nothing}=nothing, sco::Int=0)
    if scratch === nothing
        scratch = Memory{Limb}(undef, mu_divrem_scratch_len(n, m))
        sco = 0
    end
    l = leading_zeros(@inbounds d[do_+m])
    nn = n + 1
    uo = sco
    dvo = sco + nn
    ivo = sco + nn + m
    wso = sco + nn + 2m
    # The appended top limb is what makes Q < β^(nn-m), which mu_divrem_core!
    # relies on for its leading block (cf. divrem!).
    if l == 0
        copyto!(scratch, uo + 1, a, ao + 1, n)
        @inbounds scratch[uo+nn] = zero(Limb)
        dv, dvv = d, do_
    else
        @inbounds scratch[uo+nn] = lshift!(scratch, uo, a, ao, n, l)
        lshift!(scratch, dvo, d, do_, m, l)
        dv, dvv = scratch, dvo
    end
    # The top ii limbs of dv are normalized too (same top limb), so invertappr!'s
    # precondition carries over unchanged.
    ii = mu_inv_size(n - m + 1, m)
    invertappr!(scratch, ivo, dv, dvv + (m - ii), ii, scratch, wso)
    mu_divrem_core!(q, qo, scratch, uo, nn, dv, dvv, m, scratch, ivo, ii, scratch, wso)
    if l == 0
        copyto!(r, ro + 1, scratch, uo + 1, m)
    else
        rshift!(r, ro, scratch, uo, m, l)
    end
    return nothing
end

# ---- Barrett (mu) reduction ------------------------------------------------
# T mod m for a *fixed* modulus: one reciprocal built per modulus and reused
# across every product, where mu_divrem! rebuilds one per call. That is the
# shape powermod_limbs needs — ~630 reductions per setup for a 512-bit exponent
# — and the only reason a separate entry point exists at all.
#
# Two multiplications per reduction (the estimate and q̂·d), same as a
# hand-rolled Barrett (HAC 14.42), but sharing invertappr! and mu_div_step!
# with the division tier instead of carrying a second reciprocal convention.

# powermod_limbs switches its reduction to Barrett at these modulus sizes
# (limbs), tuned by bench/bench_kernels.jl barrett. The baselines differ by
# parity, so the crossovers do too: Montgomery redc! (odd m) is a single
# schoolbook addmul sweep, while divrem! (even m) rides the dc/Barrett division
# tiers and holds out longer.
#
# mu_reduce!/baseline, 512-bit exponent, by modulus limbs k:
#
#   odd   40    48    56    64    72    80    88    96    112   128   160   192
#         1.24  0.94  1.15  1.03  0.98  1.05  1.09  0.85  0.95  0.78  0.69  0.64
#
#   even  40    48    56    64    72    80    88    96    112   128   160   192
#         1.22  1.19  1.10  1.27  1.25  1.13  1.19  1.12  1.12  1.06  0.96  0.96
#         ... 224: 0.81, 256: 0.75, 384: 0.66, 512: 0.55
#
# Both values are the first size at or above which every measured point wins,
# not a fitted crossover: the odd band from 48 to 88 swings +-15% with no trend
# (the two sides step on different boundaries -- mul!'s Karatsuba/NTT admission
# against redc!'s smooth schoolbook), so nothing inside it is defensible. Noise
# here is ~4%: even k = 192 measured 0.96 and 1.00 in two runs.
#
# Both moved with the switch from the old hand-rolled Barrett to mu_reduce!:
# odd 68 -> 96 (68 sat inside the oscillating band on unmeasured ground), even
# 240 -> 160 (the cheaper invertappr! setup pulled the crossover down).
const BARRETT_THRESHOLD = 96
const BARRETT_EVEN_THRESHOLD = 160

# Per-modulus reduction state. Concrete fields (never a Union) so the caller's
# hot loop stays type-stable; the unused case is an EMPTY_LIMBS dummy.
struct MuReduce
    l::Int                  # normalizing shift of m
    mp::Memory{Limb}        # m << l: k limbs, top bit set
    iv::Memory{Limb}        # invertappr! of mp, k limbs
    q::Memory{Limb}         # discarded quotient, k limbs
    u::Memory{Limb}         # shifted numerator, 2k limbs, destroyed per call
    scratch::Memory{Limb}   # mu_div_step! working space
end

mu_reduce_empty() =
    MuReduce(0, EMPTY_LIMBS, EMPTY_LIMBS, EMPTY_LIMBS, EMPTY_LIMBS, EMPTY_LIMBS)

# Once-per-modulus setup for the k-limb m (m[k] ≠ 0, k ≥ 2). Normalizing here
# rather than per reduction is what lets invertappr! be called once: its Newton
# derivation needs a top-bit-set divisor, but the shift is a property of m
# alone.
function mu_reduce_setup(m::Memory{Limb}, mo::Int, k::Int)
    l = leading_zeros(@inbounds m[mo+k])
    mp = Memory{Limb}(undef, k)
    if l == 0
        copyto!(mp, 1, m, mo + 1, k)
    else
        lshift!(mp, 0, m, mo, k, l)
    end
    iv = Memory{Limb}(undef, k)
    invertappr!(iv, 0, mp, 0, k)
    return MuReduce(l, mp, iv, Memory{Limb}(undef, k), Memory{Limb}(undef, 2k),
                    Memory{Limb}(undef, mu_div_step_scratch_len(k)))
end

# r[1..k] = T mod m (unnormalized), T = t[to+1..to+2k].
#
# Requires T < m·β^k, which is narrower than a general T < β^2k: it is
# mu_div_step!'s U < d·β^s precondition at s = k, and it is what makes this one
# block instead of two. powermod's T = x·y with x, y < m satisfies it (T < m²
# ≤ m·β^k) — a caller with arbitrary T < β^2k must use divrem! instead.
#
# Working in the shifted domain costs two O(k) passes against the reduction's
# own 2·M(k): T' = T·2^l stays inside 2k limbs (T' < m·m' < β^2k), and
# T' mod m' = (T mod m)·2^l shifts straight back out. r must not alias t.
function mu_reduce!(r::Memory{Limb}, ro::Int, t::Memory{Limb}, to::Int, k::Int,
                    st::MuReduce)
    if st.l == 0
        copyto!(st.u, 1, t, to + 1, 2k)
    else
        lshift!(st.u, 0, t, to, 2k, st.l)
    end
    mu_div_step!(st.q, 0, st.u, 0, k, st.mp, 0, k, st.iv, 0, k, st.scratch, 0)
    if st.l == 0
        copyto!(r, ro + 1, st.u, 1, k)
    else
        rshift!(r, ro, st.u, 0, k, st.l)
    end
    return nothing
end
