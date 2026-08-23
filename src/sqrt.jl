# Karatsuba square root (Zimmermann, INRIA RR-3805). Two entry points at the
# bottom of this file: sqrtrem! (root and remainder) and sqrt! (root only).
#
# Order here is definition-before-use: tuning constants, then the scratch
# layout, then the helpers, then the two entry points.

# ---- tuning -----------------------------------------------------------------

# Root-only top level switches from exact divrem! (whose remainder doubles as
# the nonnegativity certificate) to divappr_dc! + guard-limb certificate once the
# quotient is long enough for the skipped remainder work to beat the rare
# fallback mul (bench/bench_sqrt_thr.jl: flat 4-48, gains from lq ≈ 16 up —
# 4k bits — and no measurable win below; the division there is schoolbook,
# where divappr_bc!'s triangle is the whole saving).
const SQRT_DIVAPPR_THRESHOLD = 16

# Two separate decisions; collapsing them into one constant costs time at both
# ends, since they are gated by different things.
#
# MU_SQRT_MIN — should *this* level divide with Barrett? Its reciprocal arrives
# free from the child, so this is Barrett-against-dc with the inverse already
# paid. Barrett's two products are both at the full divisor size while dc
# recurses on halves, so with the inverse free the comparison is just
# 2·M(m) vs D(m) — Barrett wins exactly when D/M > 2:
#
#   m        128    192    256    512    2048
#   D/M      1.81   1.97   2.89   4.3    6.1
#
# D/M crosses 2 right where mul! enters the fp NTT (M goes 4829 -> 5658 from
# m = 192 to 256, only +17% for +33% size; Karatsuba at 256 would be 9017), so
# this cutoff is really inherited from MUL_FPNTT_THRESHOLD.
#
# Going below 256 is free on the ladder side — the chain passes through those
# widths anyway, and inv(128) + step(256) is exactly inv(256) — but Barrett
# itself loses there: MU_SQRT_MIN = 128 measured 165.2/341.6us at 4096/8192
# limbs against 256's 159.3/335.4.
#
# MU_SQRT_LADDER — should the ladder exist at all? Its base has no seed and
# costs a from-scratch invertappr!, which only repays itself once several
# levels divide with it. That is a property of the *input* size, not of any one
# level's divisor: at a top-level divisor of 1024 the chain serves three widths;
# at 512 it serves one and loses. Gating both decisions on 256 regressed
# 1024-limb isqrt by 27%; gating both on 1024 gave up the 4096-limb win and left
# 8192 at 346us where 256 reached 336us.
#
# isqrt us, ladder against plain divappr_dc!, by top-level divisor:
#
#   hh        704    768    960    1024   2048
#   dc       100.8  117.7  179.0  171.5  387.5
#   ladder    99.1  123.7  155.7  160.1  337.7
#
# so 896. A middle tier that bought its own reciprocal for the top level was
# tried and removed: the band where it beat the ladder was one sample wide (it
# won at hh = 768, lost at 704, both inside ~2% run-to-run noise).
const MU_SQRT_MIN = 192
const MU_SQRT_LADDER = 896

# ---- scratch layout ---------------------------------------------------------

# The reciprocal ladder. Each level leaves inv(its root) — exactly the divisor
# its parent divides by — in an h-limb slot, and the child's slot nests inside
# it at ivo+lq, so the whole chain fits in the top level's h limbs.
#
#   level's slot:  iv[ivo+1 .. ivo+h]        = inv(S),        h limbs
#   child's slot:  iv[ivo+lq+1 .. ivo+h]     = inv(S_child), hh limbs
#
# The child's slot is exactly where the parent's Newton step wants its seed, so
# nothing is moved. Levels below MU_SQRT_MIN keep no reciprocal, and the
# boundary level seeds the chain with a from-scratch invertappr!.
sqrt_inv_len(h::Int) = h + 4

# Offset of the reciprocal ladder: everything the divisions need comes first,
# and the ladder sits above it so it survives the parent overwriting its own
# num/qq/dv regions (the recursion reuses those at the same sco).
#
# The last region is shared by four users, whichever is largest: the dc path's
# remainder/Q² overlay (the divisor is read from the root buffer, so it needs no
# copy slot), the Barrett block scratch, and the two ways a level can build its
# reciprocal — one ladder step, or the from-scratch base.
function sqrt_inv_base(h::Int)
    lq = h >> 1
    hh = h - lq
    return (h + 2) + (lq + 3) +
           max(h + 3 + 2hh, mu_div_step_scratch_len(hh),
               invertappr_step_scratch_len(h), invertappr_scratch_len(h))
end

# num + quotient slot + division scratch, then the reciprocal ladder.
sqrt_scratch_len(h::Int) = sqrt_inv_base(h) + sqrt_inv_len(h)

# ---- helpers ----------------------------------------------------------------

# Build the level's division numerator at scratch[num+1..]: a ×β guard limb,
# then N = (c1, R', A1) halved in place — (Q, U) = divrem(N, 2S') is run as
# ⌊(N>>1)/S'⌋ with U = 2·(N>>1 mod S') + ε, so the divisor is the normalized
# root itself (no 2S' buffer, whose carry limb would force a 63-bit
# renormalization of both operands at every level). The guard limb holds
# ε·2^63 after the halving, so the buffer is N·β/2 exactly and the divappr
# guard quotient keeps its meaning. Returns (numlen, ε); numlen ≤ h after
# the halving strips c1's bit into the limb below.
@inline function sqrt_build_num!(scratch::Memory{Limb}, num::Int, a::Memory{Limb},
                                 ao::Int, lq::Int, h::Int, hh::Int, c1::Int)
    @inbounds scratch[num+1] = zero(Limb)
    copyto!(scratch, num + 2, a, ao + lq + 1, h)
    numlen = h
    if c1 != 0
        numlen = h + 1
        @inbounds scratch[num+1+numlen] = c1 % Limb
    end
    ε = @inbounds scratch[num+2] & one(Limb)
    rshift!(scratch, num, scratch, num, numlen + 1, 1)
    @inbounds while numlen > hh && scratch[num+1+numlen] == 0
        numlen -= 1
    end
    return numlen, ε
end

# Leave inv(S) — the reciprocal the parent divides by — in scratch[ivo+1..ivo+h].
# The caller runs this after its correction loop, since that can decrement S.
#
# When the child kept a reciprocal it is already right-aligned at ivo+lq+1,
# exactly where the doubling step wants its seed. At odd h the child's hh limbs
# are already the guard width (lq+1) and one step finishes. At even h the step
# wants one limb more than the child's root: that limb is the top of *this*
# level's quotient, which did not exist until the division ran, so the widen
# belongs here and not in the child. Both steps are guarded, so the ladder holds
# ~1 ulp — seeding the doubling at exactly half instead compounds to 2.6e15 ulps
# by 2048 limbs.
function sqrt_build_inv!(s::Memory{Limb}, so::Int, h::Int, lq::Int, hh::Int,
                         scratch::Memory{Limb}, ivo::Int, wso::Int, seeded::Bool)
    if !seeded
        # ladder base: below MU_SQRT_MIN no child keeps a reciprocal
        return invertappr!(scratch, ivo, s, so, h, scratch, wso)
    end
    w = invertappr_seed_len(h)                      # = lq + 1
    if hh < w
        invertappr_step!(scratch, ivo + hh - 1, s, so + hh - 1, w, scratch, wso, hh)
    end
    invertappr_step!(scratch, ivo, s, so, h, scratch, wso, w)
    return nothing
end

# a·2^δ ≥ b, exactly, for UInt128 mantissas with a signed binary exponent gap.
@inline function ge_scaled(a::UInt128, b::UInt128, δ::Int)
    if δ >= 0
        δ >= 128 && return a != 0 || b == 0
        mask = (UInt128(1) << δ) - 1
        return a >= (b >> δ) + ((b & mask) != 0 ? UInt128(1) : UInt128(0))
    end
    sδ = -δ
    sδ >= 128 && return b == 0
    b > (typemax(UInt128) >> sδ) && return false
    return a >= (b << sδ)
end

# Certificate for the divappr sqrt path: decide sign(R), R = U·β^lq + A0 - Q²,
# without U. With Q exact and g the guard limb, floor(U·β/D) ∈ [g-E, g], so
# U ∈ [(g-E)·D/β, (g+1)·D/β); D = 2S' and Q are bracketed by their top 64
# bits (db·2^ed ≤ D < (db+1)·2^ed, likewise qb/eq) — S' is top-bit
# normalized, so db is its top limb exactly and ed carries the doubling.
# Returns 1 for R ≥ 0 certain (S is the root), -1 for R < 0 certain (root is
# S-1: R > -β^2lq ≥ -(2S-1) since 2S ≥ β^hh·β^lq, so a single decrement
# always lands), 0 for the ambiguous band around R = 0 (~2^-57 of inputs,
# plus perfect squares).
function sqrt_root_cert(s::Memory{Limb}, so::Int, lq::Int, hh::Int, g::Limb)
    # The divappr engines over-approximate by ≤ ~20 ulps (derivation at
    # divappr_dc_partial!). Kept loose at 32; sqrt_appr_top! rejects on the same
    # value, so the two must agree.
    E = Limb(32)
    db = @inbounds s[so+lq+hh]
    ed = 64 * (hh - 1) + 1
    i = lq
    @inbounds while i > 0 && s[so+i] == zero(Limb)
        i -= 1
    end
    i == 0 && return 1                    # Q = 0: R = U·β^lq + A0 ≥ 0
    q1 = @inbounds s[so+i]
    lzq = leading_zeros(q1)
    qb = lzq == 0 ? q1 :
         (q1 << lzq) | (i > 1 ? (@inbounds(s[so+i-1]) >> (64 - lzq)) : zero(Limb))
    eq = 64 * (i - 1) - lzq
    eA = ed - 64 + 64lq
    # R ≥ 0 ⟸ (g-E)·db·2^eA ≥ (qb+1)²·2^2eq > Q² (A0 ≥ 0 only helps)
    B, eB = qb == typemax(Limb) ? (UInt128(1), 2eq + 128) :
                                  (widemul(qb + one(Limb), qb + one(Limb)), 2eq)
    ge_scaled(widemul(g - E, db), B, eA - eB) && return 1
    # R < 0 ⟸ (g+1)·(db+1)·2^eA + β^lq ≤ qb²·2^2eq ≤ Q²; the A0 < β^lq slack
    # folds into one LHS ulp since β^lq ≤ 2^eA (dl ≥ 3 here). The two typemax
    # corners would overflow the mantissa product; punt them to the fallback.
    if g != typemax(Limb) && db != typemax(Limb)
        A2 = widemul(g + one(Limb), db + one(Limb))
        ge_scaled(widemul(qb, qb), A2 + UInt128(1), 2eq - eA) && return -1
    end
    return 0
end

# Top-level root-only quotient via the divappr engines: N·β ÷ 2S' with one
# guard limb ĝ, run as (N·β/2) ÷ S' on the pre-halved num buffer (guard limb
# ε·2^63) so the divisor is the normalized root itself. Returns (code, rhi):
# code 1 — certificate settled the root in s (possibly decremented); code 0 —
# retry exactly (guard wrap ĝ < E leaves the integer part uncertain, or the
# quotient overflowed β^lq; ~2^-59); code 2 — Q is exact and in s but
# sign(R) is ambiguous: U = 2·(N₁ - Q·S') + ε has been reconstructed with
# one hh×lq mul into a[ao+lq+1..] (top limb in rhi) and the caller finishes
# with the shared exact remainder phase.
function sqrt_appr_top!(s::Memory{Limb}, so::Int, a::Memory{Limb}, ao::Int,
                        h::Int, lq::Int, hh::Int, ε::Limb, c1::Int,
                        scratch::Memory{Limb}, num::Int, numlen::Int,
                        qq::Int, dv::Int, ivo::Int, barrett::Bool)
    nA = numlen + 1
    @inbounds while nA > hh && scratch[num+nA] == zero(Limb)
        nA -= 1
    end
    # N too short for a meaningful guard (needs R' = 0 and a zero A1 top —
    # e.g. a perfect-square upper half): let the exact path handle it
    @inbounds scratch[num+nA] == zero(Limb) && return 0, 0
    fill!(view(scratch, qq+1:qq+lq+3), zero(Limb))
    # run the divappr engines on the numerator buffer directly (it is
    # disposable — the rare paths that still need N rebuild it from a); the
    # appended zero top limb pins the carry-out to zero
    nn = nA + 1
    @inbounds scratch[num+nn] = zero(Limb)
    if barrett
        # the ladder already holds inv(S') at exactly this divisor's width, so
        # the top level's Barrett quotient costs no reciprocal at all. Its ≤ 6
        # ulp over-approximation is well inside the 32 the certificate allows.
        mu_divappr_core!(scratch, qq, scratch, num, nn, s, so + lq, hh,
                         scratch, ivo + lq, scratch, dv)
    else
        v = @inbounds invert_pi1(s[so+h], s[so+h-1])
        if hh >= DC_DIV_THRESHOLD && nn - hh >= DC_DIV_THRESHOLD
            divappr_dc!(scratch, qq, scratch, num, nn, s, so + lq, hh, v,
                        DC_DIV_THRESHOLD, scratch, dv)
        else
            divappr_bc!(scratch, qq, scratch, num, nn, s, so + lq, hh, v)
        end
    end
    qext = nA - hh + 1               # written limbs: guard, then integer part
    @inbounds for i in qq+lq+2:qq+qext
        scratch[i] != zero(Limb) && return 0, 0
    end
    g = @inbounds scratch[qq+1]
    g < Limb(32) && return 0, 0      # divappr error bound, per sqrt_root_cert
    copyto!(s, so + 1, scratch, qq + 2, lq)
    verdict = sqrt_root_cert(s, so, lq, hh, g)
    if verdict != 0
        verdict < 0 && sub_1!(s, so, s, so, h, one(Limb))
        return 1, 0
    end
    numlen, ε = sqrt_build_num!(scratch, num, a, ao, lq, h, hh, c1)
    mul!(scratch, dv, s, so + lq, hh, scratch, qq + 1, lq)   # Q·S' ≤ N₁
    plen = hh + lq
    @inbounds while plen > numlen && scratch[dv+plen] == zero(Limb)
        plen -= 1
    end
    sub!(scratch, num + 1, scratch, num + 1, numlen, scratch, dv, plen)
    uc = lshift!(a, ao + lq, scratch, num + 1, hh, 1)        # U = 2U₁ + ε
    @inbounds a[ao+lq+1] |= ε
    return 2, Int(uc)
end

# ---- entry points -----------------------------------------------------------

# s[1..h] = isqrt(a[1..n]), h = (n+1)>>1. Requires n even (or n <= 2) and
# a[ao+n] >= 2^62 (caller normalizes by an even bit shift plus a zero low limb
# for odd lengths); an odd-length high part would leave S' half-normalized,
# 2S' < β^hh, and the quotient step unbounded.
#
# On return a[ao+1..ao+h] holds the low limbs of the remainder a - s^2; the
# return value is its top limb (0 or 1). scratch needs sqrt_scratch_len(h)
# limbs at sco; recursion levels share it (a level touches it only after its
# child returns).
#
# With needrem = false only the root is guaranteed (a is still destroyed and
# the return value is meaningless): above SQRT_DIVAPPR_THRESHOLD the top-level
# division runs through the divappr engines (no remainder computed) and a
# guard-limb certificate settles sign(R), R = A - S², outright — U is
# reconstructed with one mul only in the ambiguous band (~2^-57 of inputs, plus
# perfect squares). Below the threshold the exact division's remainder feeds
# cheap positivity checks, and the final remainder phase — a quarter-size Q²
# square plus an h-limb subtract — runs only when those can't fire. The
# recursion always needs the child's remainder, so only the top level skips.
#
# ivo is the base of this level's reciprocal slot (see sqrt_inv_len); wantinv
# says whether the caller will divide by this level's root with Barrett and so
# needs inv(S) left there. ivo < 0 means no ladder at all.
function sqrtrem!(s::Memory{Limb}, so::Int, a::Memory{Limb}, ao::Int, n::Int,
                  scratch::Memory{Limb}, sco::Int=0, needrem::Bool=true,
                  ivo::Int=-1, wantinv::Bool=false)
    if n <= 2
        v = n == 1 ? UInt128(@inbounds a[ao+1]) :
            (UInt128(@inbounds a[ao+2]) << 64) | (@inbounds a[ao+1])
        rt = isqrt(v)
        rm = v - rt * rt
        @inbounds s[so+1] = rt % Limb
        @inbounds a[ao+1] = rm % Limb
        return Int((rm >> 64) % Limb)
    end
    h = (n + 1) >> 1
    lq = h >> 1       # low-root limbs (the quotient Q)
    hh = h - lq       # high-root limbs (S')
    nh = n - 2lq      # limbs of the high part
    # a = Ahi*β^2lq + A1*β^lq + A0; recurse: Ahi = S'^2 + R', R' <= 2S'.
    # A Barrett division here needs the child's reciprocal, which the child
    # produces iff its own root reaches the threshold — and its root is exactly
    # this level's divisor, so the two conditions are the same test.
    barrett = ivo >= 0 && hh >= MU_SQRT_MIN
    c1 = sqrtrem!(s, so + lq, a, ao + 2lq, nh, scratch, sco, true,
                  ivo < 0 ? -1 : ivo + lq, barrett)
    num = sco                    # h+2 limbs: ×β guard, then (c1, R', A1)/2
    qq = sco + h + 2             # lq+3 limbs: quotient Q (divappr: ĝ below Q)
    dv = sco + h + lq + 5        # ≤ h+3+2hh limbs: divrem!/divappr_dc! scratch
                                 # (the divisor S' is normalized, so its copy
                                 # slot inside the contract is never touched);
                                 # doubles as U's slot — divrem! leaves the
                                 # remainder at its scratch base, so passing
                                 # (r, ro) = (scratch, dv) makes the final
                                 # copy a no-op — and as Q²/Q·S' space once
                                 # the division is over
    numlen, ε = sqrt_build_num!(scratch, num, a, ao, lq, h, hh, c1)
    rhi = 0
    appr = 0
    if !needrem && lq >= SQRT_DIVAPPR_THRESHOLD
        appr, rhi = sqrt_appr_top!(s, so, a, ao, h, lq, hh, ε, c1,
                                   scratch, num, numlen, qq, dv, ivo, barrett)
        appr == 1 && return 0    # certificate settled the root
        # the engines destroyed the numerator; rebuild it for the exact retry
        appr == 0 && (numlen, ε = sqrt_build_num!(scratch, num, a, ao, lq, h, hh, c1))
    end
    if appr == 0
        # Pre-zero the quotient slot so untouched high limbs read as zero.
        fill!(view(scratch, qq+1:qq+lq+2), zero(Limb))
        qlen = numlen - hh + 1
        if hh >= 2
            # The numerator buffer is disposable and S' normalized, so run
            # the division engines on it directly — no defensive copy, no
            # entry dispatch, remainder left in place. The appended zero top
            # limb pins the extra quotient bit to zero (Q < β^qlen).
            nn = numlen + 1
            @inbounds scratch[num+1+nn] = zero(Limb)
            if barrett
                # the child left inv(S') at ivo+lq, at exactly the divisor's
                # width — no reciprocal work at all on this path
                mu_divrem_core!(scratch, qq, scratch, num + 1, nn, s, so + lq, hh,
                                scratch, ivo + lq, hh, scratch, dv)
            else
                v = @inbounds invert_pi1(s[so+h], s[so+h-1])
                if hh >= DC_DIV_THRESHOLD && nn - hh >= DC_DIV_THRESHOLD
                    divrem_dc!(scratch, qq, scratch, num + 1, nn, s, so + lq, hh, v,
                               DC_DIV_THRESHOLD, scratch, dv)
                else
                    divrem_bc!(scratch, qq, scratch, num + 1, nn, s, so + lq, hh, v)
                end
            end
            uc = lshift!(a, ao + lq, scratch, num + 1, hh, 1)  # U = 2U₁+ε < 2S'
        else
            divrem!(scratch, qq, scratch, dv, scratch, num + 1, numlen,
                    s, so + lq, hh, scratch, dv)
            uc = lshift!(a, ao + lq, scratch, dv, hh, 1)
        end
        @inbounds a[ao+lq+1] |= ε
        rhi = Int(uc)
        # Q <= β^lq; if Q = β^lq exactly, clamp to β^lq - 1 and put 2S' back in U
        # (still >= the true root; the correction loop repairs the remainder).
        toobig = any(!iszero, view(scratch, qq+lq+1:qq+qlen))
        if toobig
            fill!(view(scratch, qq+1:qq+lq), typemax(Limb))
            rhi += Int(add_n!(a, ao + lq, a, ao + lq, s, so + lq, hh))
            rhi += Int(add_n!(a, ao + lq, a, ao + lq, s, so + lq, hh))
        end
        copyto!(s, so + 1, scratch, qq + 1, lq)
    end
    if !needrem
        # S never undershoots (the correction loop below only decrements),
        # so S is exact iff R = V - Q² ≥ 0 with V = rhi·β^h + U·β^lq + A0.
        # Provably nonnegative when V ≥ β^2lq > Q²: rhi ≠ 0, U ≥ β^lq, or
        # V's top two limbs clear (q1+1)² ≥ Q²/β^(2lq-2). Otherwise fall
        # through and settle it exactly.
        rhi != 0 && return 0
        normlen(a, ao + 2lq, h - 2lq) != 0 && return 0
        q1 = @inbounds s[so+lq]
        if q1 != typemax(Limb)
            v = (UInt128(@inbounds a[ao+2lq]) << 64) | (@inbounds a[ao+2lq-1])
            widemul(q1 + one(Limb), q1 + one(Limb)) <= v && return 0
        end
    end
    # R = U*β^lq + A0 - Q^2, tracked as (rhi, a[ao+1..ao+h]) with rhi signed
    sqr!(scratch, dv, s, so, lq)
    rhi -= Int(sub!(a, ao, a, ao, h, scratch, dv, 2lq))
    while rhi < 0
        # (S+1)^2 overshoots: R += 2S - 1, S -= 1
        tc = lshift!(scratch, num, s, so, h, 1)
        sub_1!(scratch, num, scratch, num, h, one(Limb))
        c = add_n!(a, ao, a, ao, scratch, num, h)
        rhi += Int(c) + Int(tc)
        sub_1!(s, so, s, so, h, one(Limb))
    end
    wantinv && sqrt_build_inv!(s, so, h, lq, hh, scratch, ivo, dv, barrett)
    return rhi
end

# Root-only square root: sqrtrem! with the top-level remainder phase elided;
# same contract, but a's contents on return are unspecified.
function sqrt!(s::Memory{Limb}, so::Int, a::Memory{Limb}, ao::Int, n::Int,
               scratch::Memory{Limb}, sco::Int=0)
    h = (n + 1) >> 1
    # Turn the ladder on from the input size, not from any level's divisor —
    # see MU_SQRT_LADDER. h - (h>>1) is the top level's divisor width.
    ivo = h - (h >> 1) >= MU_SQRT_LADDER ? sco + sqrt_inv_base(h) : -1
    sqrtrem!(s, so, a, ao, n, scratch, sco, false, ivo, false)
    return nothing
end
