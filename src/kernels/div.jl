# Division kernels: reciprocals and schoolbook divrem (mpn layer).

# Möller–Granlund reciprocal: v = ⌊(β²-1)/d⌋ - β for normalized d (top bit set).
# One hardware 128/64 divide at setup; every per-limb divide becomes 2 muls.
@inline function invert_limb(d::Limb)
    return ((typemax(DLimb) - (DLimb(d) << 64)) ÷ d) % Limb
end

# Rare second-fixup bodies. @noinline keeps them as real (predicted-not-taken)
# branches: inlined, LLVM if-converts them to cmovs on the loop-carried
# remainder chain, costing ~3 cycles per limb in the divrem_1!/divrem_2! loops.
@noinline div_fixup(q1::Limb, r::Integer, d::Integer) = (q1 + one(Limb), r - d)

# Divide (u1:u0) by normalized d given v = invert_limb(d); requires u1 < d.
# The first fixup fires ~50% of the time, so it is masked (branch-free) to keep
# the loop-carried remainder chain free of mispredicts; the second is rare and branchy.
@inline function div_2by1(u1::Limb, u0::Limb, d::Limb, v::Limb)
    p = DLimb(v) * u1 + ((DLimb(u1) << 64) | u0)
    q1 = (p >> 64) % Limb + one(Limb)
    q0 = p % Limb
    r = u0 - q1 * d
    mask = -Limb(r > q0)
    q1 += mask
    r += d & mask
    if r >= d
        q1, r = div_fixup(q1, r, d)
    end
    return q1, r
end

# 3/2 reciprocal (Möller–Granlund Alg. 6): v = ⌊(β³-1)/(d1·β+d0)⌋ - β,
# for normalized d1; refines invert_limb(d1) by two conditional corrections.
@inline function invert_pi1(d1::Limb, d0::Limb)
    v = invert_limb(d1)
    p = d1 * v
    p += d0
    if p < d0
        v -= one(Limb)
        if p >= d1
            v -= one(Limb)
            p -= d1
        end
        p -= d1
    end
    t = DLimb(v) * d0
    t1 = (t >> 64) % Limb
    t0 = t % Limb
    p += t1
    if p < t1
        v -= one(Limb)
        if (p > d1) | ((p == d1) & (t0 >= d0))
            v -= one(Limb)
        end
    end
    return v
end

# Divide ⟨u2,u1,u0⟩ by normalized ⟨d1,d0⟩ given v = invert_pi1(d1, d0);
# requires ⟨u2,u1⟩ < ⟨d1,d0⟩. Returns (q, r1, r0) with ⟨r1,r0⟩ the remainder.
# Möller–Granlund Alg. 2: candidate off by at most 1 either way, same fixup
# structure as div_2by1 (first masked, second rare and branchy).
@inline function div_3by2(u2::Limb, u1::Limb, u0::Limb, d1::Limb, d0::Limb, v::Limb)
    dd = (DLimb(d1) << 64) | d0
    q = DLimb(v) * u2 + ((DLimb(u2) << 64) | u1)
    q1 = (q >> 64) % Limb
    q0 = q % Limb
    # r = ⟨u1,u0⟩ - q1*⟨d1,d0⟩ - dd (mod β²), regrouped so the dd subtraction
    # runs off the q1 critical path and both q1 muls merge into low128(q1*dd)
    w = ((DLimb(u1) << 64) | u0) - dd
    r = w - (DLimb(q1) * d0 + (DLimb(q1 * d1) << 64))
    q1 += one(Limb)
    # first fixup fires ~50%: masked to avoid mispredicts on the remainder chain
    mask = -Limb((r >> 64) % Limb >= q0)
    q1 += mask
    r += dd & ((DLimb(mask) << 64) | mask)
    if r >= dd
        q1, r = div_fixup(q1, r, dd)
    end
    return q1, (r >> 64) % Limb, r % Limb
end

# Two dividend limbs per iteration with a lazy (unreduced, two-limb) running
# remainder R < β²: per pair, R ← r1·B3 + r0·B2 + ⟨u1,u0⟩ folded mod β²
# (B2 = β² mod d, B3 = β³ mod d), so the loop-carried chain is two independent
# muls plus adds — no 2/1 division inside the loop. Quotient mass accrues off
# the chain via K2 = ⌊β²/d⌋ = β+v, K3 = ⌊β³/d⌋ = β·K2 + q3 into a sliding
# two-limb pending window ⟨p1,p0⟩; the finalized limbs written each iteration
# cannot receive later carries (all future pieces land strictly below), so no
# ripple into written quotient limbs is needed. Requires d normalized,
# d ≠ 2^63 (K2 = 2β wouldn't fit the k2lo limb), n ≥ 4.
function divrem_1_pi2!(q::Memory{Limb}, qo::Int, a::Memory{Limb}, ao::Int, n::Int, d::Limb, v::Limb)
    # setup: no hardware divide — B2 = β² - K2·d via wrapping negation
    b2 = (-(widemul(d, v) + (DLimb(d) << 64))) % Limb
    q3, b3 = div_2by1(b2, zero(Limb), d, v)     # ⟨B2,0⟩ = q3·d + B3
    k3l = q3                                    # K3 - β² = ⟨v, q3⟩
    r1 = @inbounds a[ao+n]
    r0 = @inbounds a[ao+n-1]
    p1 = zero(Limb)
    p0 = zero(Limb)
    i = n - 3
    @inbounds while i >= 1
        u1 = a[ao+i+1]
        u0 = a[ao+i]
        # remainder chain: W = r1·B3 + r0·B2 + ⟨u1,u0⟩ (≤ 130 bits), then fold
        # the c ∈ {0,1,2} overflows of weight β² back in as c·B2
        m3 = widemul(r1, b3)
        m2 = widemul(r0, b2)
        w, o1 = Base.add_with_overflow(m3, m2)
        w, o2 = Base.add_with_overflow(w, (DLimb(u1) << 64) | u0)
        c = Limb(o1) + Limb(o2)
        # c·B2 as two masked adds keeps a mul off the loop-carried chain
        w, o3 = Base.add_with_overflow(w, DLimb(b2 & (-Limb(o1))) + (b2 & (-Limb(o2))))
        if o3                                    # rare second fold; w < 3β now
            w += b2
            c += one(Limb)
        end
        # quotient piece Qp = r1·K3 + (r0+c)·K2 < 2β³ at weight β^(i-1) with
        # K3 = β² + ⟨k3h,k3l⟩, K2 = β + v. f = r0+c may wrap: the missing β·K2
        # contributes fc·v at weight β¹ and fc at weight β². By weight:
        #   L0: lo(r1·k3l), lo(f·v)
        #   L1: hi(r1·k3l), hi(f·v), lo(r1·k3h), f, fc·v
        #   L2: hi(r1·k3h), r1, fc, p0   (+ carries)
        #   L3: p1                        (+ carries; no carry-out: S is final)
        f = r0 + c
        fc = Limb(f < r0)
        ml = widemul(r1, k3l)
        mh = widemul(r1, v)                      # k3h == v
        A, cA = Base.add_with_overflow(ml, widemul(f, v))
        A, cA2 = Base.add_with_overflow(A, DLimb(v & (-fc)) << 64)
        s0 = A % Limb
        B = (A >> 64) + (mh % Limb) + f
        s1 = B % Limb
        C = (mh >> 64) + r1 + fc + p0 +
            (B >> 64) + Limb(cA) + Limb(cA2)
        s2 = C % Limb
        s3 = p1 + ((C >> 64) % Limb)   # no wrap: finalized mass < β²
        q[qo+i+3] = s3
        q[qo+i+2] = s2
        p1 = s1
        p0 = s0
        r1 = (w >> 64) % Limb
        r0 = w % Limb
        i -= 2
    end
    if i == 0   # one leftover limb: V = R·β + u[1], quotient at weight β⁰
        ul = @inbounds a[ao+1]
        m2 = widemul(r1, b2)
        w, o1 = Base.add_with_overflow(m2, (DLimb(r0) << 64) | ul)
        c = Limb(o1)
        w, o2 = Base.add_with_overflow(w, DLimb(c) * b2)
        if o2
            w += b2
            c += one(Limb)
        end
        x1 = (w >> 64) % Limb
        x0 = w % Limb
        b = x1 >= d
        b && (x1 -= d)
        q2, rem = div_2by1(x1, x0, d, v)
        # Qt = (r1+c)·K2 + b·β + q2; pending sits at (q[3], q[2])
        f = r1 + c
        fc = Limb(f < r1)
        A, cA = Base.add_with_overflow(widemul(f, v), DLimb(q2))
        A, cA2 = Base.add_with_overflow(A, DLimb(v & (-fc)) << 64)
        B = (A >> 64) + f + Limb(b) + p0
        C = (B >> 64) + fc + p1 + Limb(cA) + Limb(cA2)
        @inbounds q[qo+1] = A % Limb
        @inbounds q[qo+2] = B % Limb
        @inbounds q[qo+3] = C % Limb   # no wrap: finalized mass < β
        return rem
    end
    # no leftover: reduce R = ⟨r1,r0⟩, quotient Qt = b·β + q2 into ⟨p1,p0⟩
    b = r1 >= d
    b && (r1 -= d)
    q2, rem = div_2by1(r1, r0, d, v)
    s = DLimb(p0) + q2
    @inbounds q[qo+1] = s % Limb
    @inbounds q[qo+2] = p1 + ((s >> 64) % Limb) + Limb(b)
    return rem
end

function divrem_1!(q::Memory{Limb}, qo::Int, a::Memory{Limb}, ao::Int, n::Int, d::Limb)
    l = leading_zeros(d)
    dn = d << l
    v = invert_limb(dn)
    if l == 0
        # below ~10 limbs the pi2 setup (one div_2by1 + muls) outweighs the
        # faster loop; benchmark-tuned crossover
        if n >= 10 && d != (one(Limb) << 63)
            return divrem_1_pi2!(q, qo, a, ao, n, d, v)
        end
        rem = zero(Limb)
        @inbounds for i in n:-1:1
            q[qo+i], rem = div_2by1(rem, a[ao+i], d, v)
        end
        return rem
    end
    # Divide the left-shifted numerator by dn; quotient limbs are unchanged
    # and the true remainder is rem >> l. The spill a[n] >> (64-l) < 2^l ≤ dn
    # seeds the running remainder, so no extra quotient limb is needed.
    rem = @inbounds a[ao+n] >> (64 - l)
    @inbounds for i in n:-1:2
        u = (a[ao+i] << l) | (a[ao+i-1] >> (64 - l))
        q[qo+i], rem = div_2by1(rem, u, dn, v)
    end
    @inbounds q[qo+1], rem = div_2by1(rem, a[ao+1] << l, dn, v)
    return rem >> l
end

# a[1..n] ÷ ⟨d1,d0⟩ (d1 ≠ 0, n ≥ 2): writes n-1 quotient limbs (top may be
# zero), returns the remainder (r1, r0). The two-limb remainder window stays in
# registers and unnormalized divisors are handled by shifting the numerator on
# the fly, so no scratch or numerator copy is needed (cf. mpn_divrem_2).
# q may alias a at the same offset.
function divrem_2!(q::Memory{Limb}, qo::Int, a::Memory{Limb}, ao::Int, n::Int, d1::Limb, d0::Limb)
    l = leading_zeros(d1)
    if l == 0
        v = invert_pi1(d1, d0)
        dd = (DLimb(d1) << 64) | d0
        w = @inbounds (DLimb(a[ao+n]) << 64) | a[ao+n-1]
        qh = w >= dd
        qh && (w -= dd)
        @inbounds q[qo+n-1] = qh
        r1 = (w >> 64) % Limb
        r0 = w % Limb
        @inbounds for j in n-2:-1:1
            qhat, r1, r0 = div_3by2(r1, r0, a[ao+j], d1, d0, v)
            q[qo+j] = qhat
        end
        return r1, r0
    end
    # shifted numerator has n+1 limbs; its top spill a[n] >> (64-l) < 2^l ≤ dn1
    # seeds the window, so exactly n-1 quotient limbs come out of the loop
    dn1 = (d1 << l) | (d0 >> (64 - l))
    dn0 = d0 << l
    v = invert_pi1(dn1, dn0)
    r1 = @inbounds a[ao+n] >> (64 - l)
    r0 = @inbounds (a[ao+n] << l) | (a[ao+n-1] >> (64 - l))
    @inbounds for j in n-1:-1:2
        u = (a[ao+j] << l) | (a[ao+j-1] >> (64 - l))
        qhat, r1, r0 = div_3by2(r1, r0, u, dn1, dn0, v)
        q[qo+j] = qhat
    end
    qhat, r1, r0 = div_3by2(r1, r0, @inbounds(a[ao+1]) << l, dn1, dn0, v)
    @inbounds q[qo+1] = qhat
    return r1 >> l, (r0 >> l) | (r1 << (64 - l))
end

# Degenerate super-row for divrem_bc!: ⟨n1,n0⟩ == ⟨d1,d0⟩ violates the div_3by2
# precondition but forces qhat = β-1 exactly, and the window bound W < β·d rules
# out a borrow past the top limb. Refresh the stale top slot, one scalar submul,
# then return the re-paired window ⟨n1,n0⟩. Rare (cold) path. s low divisor
# limbs are skipped (0 in exact mode; approx mode's row truncation).
@inline function divrem_bc_degrow!(u::Memory{Limb}, uo::Int, j::Int,
                                   d::Memory{Limb}, do_::Int, m::Int, n0::Limb,
                                   s::Int)
    @inbounds u[uo+j+m-1] = n0
    submul_1!(u, uo+j-1+s, d, do_+s, m-s, typemax(Limb))
    return (@inbounds u[uo+j+m-1]), (@inbounds u[uo+j+m-2])
end

# Schoolbook (Knuth Algorithm D) quotient/remainder, two quotient limbs per
# pass (radix β²). u (nn limbs) is destroyed: the m-limb remainder is left in
# u[1..m] (meaningless in approx mode). d must be normalized (top bit of d[m]
# set), m ≥ 2, v = invert_pi1 of its top two limbs. Writes q[1..nn-m]; returns
# the extra top quotient bit qh.
#
# Each super-row: a 4/2 division of the top window (two chained div_3by2)
# yields the exact quotient ⟨qhi,qlo⟩ of the top four limbs by ⟨d1,d0⟩ — a
# single-super-digit Knuth estimate, at most 2 too large for normalized d
# (TAOCP 4.3.1 Thm A/B) — then one submul_2! sweep subtracts both rows at
# once, halving the passes over u versus limb-at-a-time. The top two window
# limbs ⟨n1, n0⟩ live in registers across rows (their memory slots are stale).
#
# approx mode skips each row's s lowest divisor limbs — the triangle instead
# of the square, ~qn²/2 submul work in the balanced case, keeping the top
# (row + 2) of d[1..m-2] — making the quotient a one-sided over-approximation:
# q_true ≤ q̂ ≤ q_true + 2. Soundness: row j's neglected products are confined
# to positions ≤ m-4-G (G = 2 guard limbs, constant across rows since the kept
# length shrinks with j) and total less than qn·β^(m-2-G)/2 ≤ D·β^(-G)/2,
# while every window read sits at positions ≥ m-2 — so the run is exact
# schoolbook on a numerator perturbed upward by Δ < D, and each row's divisor
# is a truncation of d (only ever smaller): the quotient never undershoots and
# overshoots by at most ⌈Δ/D⌉ + 1 ≤ 2. Add-backs must restore exactly the
# truncated window that was subtracted. The exact mode is s ≡ 0.
function divrem_bc!(q::Memory{Limb}, qo::Int, u::Memory{Limb}, uo::Int, nn::Int,
                    d::Memory{Limb}, do_::Int, m::Int, v::Limb, approx::Bool=false)
    qn = nn - m
    qh = zero(Limb)
    if cmp_limbs(u, uo+qn, m, d, do_, m) >= 0
        qh = one(Limb)
        sub_n!(u, uo+qn, u, uo+qn, d, do_, m)
    end
    d1 = @inbounds d[do_+m]
    d0 = @inbounds d[do_+m-1]
    dd = (DLimb(d1) << 64) | d0
    n1 = @inbounds u[uo+nn]
    n0 = @inbounds u[uo+nn-1]
    j = qn
    @inbounds while j >= 2
        s = approx ? max(m - 4 - j, 0) : 0
        if n1 == d1 && n0 == d0
            n1, n0 = divrem_bc_degrow!(u, uo, j, d, do_, m, n0, s)
            q[qo+j] = typemax(Limb)
            j -= 1
            continue
        end
        qhi, t1, t0 = div_3by2(n1, n0, u[uo+j+m-2], d1, d0, v)
        qlo, r1, r0 = div_3by2(t1, t0, u[uo+j+m-3], d1, d0, v)
        co1, co0 = m - 2 > s ? submul_2!(u, uo+j-2+s, d, do_+s, m-2-s, qlo, qhi) :
                               (zero(Limb), zero(Limb))
        rr = (DLimb(r1) << 64) | r0
        co = (DLimb(co1) << 64) | co0
        brw = rr < co
        rr -= co
        qq = (DLimb(qhi) << 64) | qlo
        while brw   # rare: estimate 1 or 2 too large, add the (kept) divisor back
            qq -= one(DLimb)
            c = m - 2 > s ? add_n!(u, uo+j-2+s, u, uo+j-2+s, d, do_+s, m-2-s) : zero(Limb)
            sm, o1 = Base.add_with_overflow(rr, dd)
            sm, o2 = Base.add_with_overflow(sm, DLimb(c))
            brw = !(o1 | o2)   # 128-bit overflow cancels the borrow
            rr = sm
        end
        q[qo+j] = (qq >> 64) % Limb
        q[qo+j-1] = qq % Limb
        n1 = (rr >> 64) % Limb
        n0 = rr % Limb
        j -= 2
    end
    @inbounds if j == 1   # leftover scalar row (3/2 qhat, error ≤ 1)
        s = approx ? max(m - 5, 0) : 0
        if n1 == d1 && n0 == d0
            qhat = typemax(Limb)
            n1, n0 = divrem_bc_degrow!(u, uo, 1, d, do_, m, n0, s)
        else
            qhat, r1, r0 = div_3by2(n1, n0, u[uo+m-1], d1, d0, v)
            cy = m - 2 > s ? submul_1!(u, uo+s, d, do_+s, m-2-s, qhat) : zero(Limb)
            cy1 = Limb(r0 < cy)
            r0 -= cy
            cy2 = r1 < cy1
            r1 -= cy1
            if cy2   # rare: qhat one too large, add the (kept) divisor back
                qhat -= one(Limb)
                c = m - 2 > s ? add_n!(u, uo+s, u, uo+s, d, do_+s, m-2-s) : zero(Limb)
                sm = ((DLimb(r1) << 64) | r0) + dd + c
                r1 = (sm >> 64) % Limb   # 128-bit overflow cancels the borrow
                r0 = sm % Limb
            end
            n1 = r1
            n0 = r0
        end
        q[qo+1] = qhat
    end
    @inbounds u[uo+m] = n1
    @inbounds u[uo+m-1] = n0
    return qh
end
