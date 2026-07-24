# Floating-point NTT multiplication following FLINT's fft_small design:
# coefficients live in Float64, products use the FMA error-free transform,
# and quotients use the magic-constant round.  The operands are split into
# b-bit chunks and the convolution is computed by CRT over two just-under-2^50
# primes (working modulus p1·p2 ≈ 2^99.99, so b ≈ 46 at typical sizes,
# shrinking slowly with operand length — see fp_ntt_params).
#
# Arithmetic model.  Values are exact integers in balanced representation,
# |x| < a few p.  Every operation is exactly correct (not approximately):
#   - a ± b is exact while |a ± b| < 2^53.
#   - fp_mulmod(x, w[, wpinv], F) returns r ≡ x·w (mod p) exactly:
#     h + l == x·w exactly (FMA error-free transform), q is an integer
#     within (1/2 + |x·w|·2^-52/p) of x·w/p, and r = x·w - q·p is an
#     integer small enough to be exact through the final fma and add.
#   - fp_round(v) = (v + 1.5·2^52) - 1.5·2^52 rounds exactly to the nearest
#     integer for |v| <= 2^51.
#
# A transform of length N = m·2^k (m | 315, 2^k | p - 1) is a flat list of
# uniform FpNttStage passes — the odd radices (7/5/3, radix 3 up to twice) first, then the
# radix-4 tower, plus a twiddle-free radix-2 pass when k is odd.  The reverse
# transform (fp_ntt_rev!) runs the same stages in reverse order; each stage's
# transpose reuses its forward tables (the DFT matrices are symmetric), so
# the whole network computes the same DFT.  Fed the forward-scrambled
# pointwise product it returns N·c[(N-t) mod N] in natural order; the unpack
# reads descending and folds in the 1/N.
#
# Lazy bounds, at p < 2^50 (write κ = p·2^-52 <= 1/4; a constant mulmod on
# |x| <= c·p returns <= p(1/2 + c·κ/2) since all stored tables are minimal
# balanced, |w| <= p/2).  One fp_reduce per butterfly closes the per-stage
# ranges (the ω^0 lane, which skips the twiddle mulmod, is reduced explicitly
# — see the butterfly comments for the per-radix numbers).  The global
# invariant: forward values converge to <= ~1.0p (so pointwise products stay
# <= p², round argument <= p), reverse raw stores to <= ~3.3p, every constant
# mulmod's round argument to <= (4p)·(p/2)/p = 2p < 2^51, and every
# intermediate sum to <= ~4.2p < 2^53/p = 8p.  The minimal-balance clamp on
# the tables is load-bearing: float-generated constants drift to ~0.8p, and
# 3.6p·0.8p breaks the 2^51·p round domain.  The differential and adversarial
# max-magnitude tests validate the ranges empirically.
#
# All bounds hold for any prime p < 2^50, so the transform is parametrized by
# an FpCtx carrying the per-prime constants and both residue transforms run
# through the same code.  Only the unpack's Garner step is two-prime specific.
#
# Hardware FMA and round-to-nearest are assumed.  No @fastmath anywhere near
# this file: contraction would break the fma(x, w, -h) cancellation.

const VF8 = SIMD.Vec{8,Float64}

# smallest small generator of GF(p)*, given the prime factors of p-1
function fp_generator(p::UInt64, facs)
    for g in 3:63
        all(powermod(UInt64(g), (p - 1) ÷ q, p) != 1 for (q, _) in facs) &&
            return UInt64(g)
    end
    error("unreachable: GF(p) has a small generator")
end

# A prime to do an NTT over, its 2-adicity, and a primitive root of GF(P).  The
# constructor checks its (prime => exponent) factorization of P-1 is complete and
# supplies the 2/3/5/7 radices, so every valid N | P-1 has a primitive Nth root —
# build_fp_plan trusts gen without rechecking order.
struct FpCtx{P}
    k::Int  # 2-adicity of P-1
    gen::UInt64
    function FpCtx{P}(facs::Tuple{Vararg{Pair{Int,Int}}}) where {P}
        @assert prod(q^e for (q, e) in facs) == P - 1
        @assert all(in(map(first, facs)), (2, 3, 5, 7))
        new{P}(trailing_zeros(P - 1), fp_generator(UInt64(P), facs))
    end
end
@inline fp_prime(::FpCtx{P}) where {P} = UInt64(P)
@inline two_adicity(F::FpCtx) = F.k

# Both primes just under 2^50 (rounding bounds above), maximizing the CRT
# modulus p1·p2, with 105·2^k | p-1 for the radix 2^k and radix 3, 5, 7
# transforms
const FP_CTX1 = FpCtx{65205 * 2^34 + 1}((2 => 34, 3 => 4, 5 => 1, 7 => 1, 23 => 1))
const FP_CTX2 = FpCtx{4095 * 2^38 + 1}((2 => 38, 3 => 2, 5 => 1, 7 => 1, 13 => 1))

# Garner constant p1^-1 mod p2 and its mulmod companion: a function of the primes
# alone, so it is not rebuilt per multiplication.
const FP_GARNER_G = Float64(invmod(fp_prime(FP_CTX1) % fp_prime(FP_CTX2), fp_prime(FP_CTX2)))
const FP_GARNER_GP = FP_GARNER_G / fp_prime(FP_CTX2)

const FP_MAGIC = 1.5 * 2^52
# exact round-to-nearest-integer for |v| <= 2^51 (scalar and VF8)
@inline fp_round(v) = (v + FP_MAGIC) - FP_MAGIC

# r ≡ x (mod p) in balanced form. Exact for any exact-integer x in the domain |x| <= 2^51·p.
# The round argument |x·inv(P)| <= 2^51, so q = fp_round(..) is valid
# and r = x - q·p is a small integer |r| <= p(1/2 + |x|·2^-52/p) < p
@inline function fp_reduce(x, ::FpCtx{P}) where {P}
    q = fp_round(x * inv(P))
    return fma(q, -P, x)
end

# r ≡ x·w (mod p) in balanced form, for exact-integer operands with
# |x·w| <= 2^51·p (so fp_round's quotient stays in range);
# then |r| <= p(1/2 + |x·w|·2^-52/p) < p.  Splitting x·w into the high part h
# and the exact FMA error l = x·w - h keeps every step exact: either both are
# integers (|x·w| >= 2^53) or h = x·w and l = 0.
# 4-arg form: w carries a precomputed wpinv = fl(w/p), so the two multiplies
# issue together (better latency, equal throughput).
@inline function fp_mulmod(x, y, F::FpCtx)
    h = x * y
    l = fma(x, y, -h)
    return fp_reduce(h, F) + l
end
@inline function fp_mulmod(x, w, wpinv, ::FpCtx{P}) where {P}
    h = x * w
    l = fma(x, w, -h)
    q = fp_round(x * wpinv)
    return fma(q, -P, h) + l
end

# Exact minimal balanced form |x| <= p/2 of a balanced integer |x| < ~p:
# the ±p shift is exact (integers below 2^53).  Stored constants must be
# minimal, not just balanced: the reverse transform feeds raw values up to
# ~3.6p into constant mulmods, and the fp_round domain |x·w| <= 2^51·p only
# holds at p -> 2^50 when |w| <= p/2.
@inline fp_minbal(x, ::FpCtx{P}) where {P} =
    ifelse(x > P / 2, x - P, ifelse(x < -P / 2, x + P, x))
@inline fp_minbal(x::VF8, ::FpCtx{P}) where {P} =
    SIMD.vifelse(x > P / 2, x - P, SIMD.vifelse(x < -P / 2, x + P, x))

# canonical form 0 <= x < p of a balanced residue (scalar and VF8)
@inline fp_canonical(x, ::FpCtx{P}) where {P} = ifelse(x < 0, x + P, x)
@inline fp_canonical(x::VF8, ::FpCtx{P}) where {P} = SIMD.vifelse(x < 0, x + P, x)

# Winograd 5-point constants from a primitive 5th root c.
# With a = c + c^4, b = c^2 + c^3 (a + b = -1, the roots sum to zero):
# k1 = -1/4,  k2 = (a - b)/4,  k3 = (c - c^4)/2,  k4 = (c^2 - c^3)/2,
# all balanced residues mod p (see fp_dft5 for the combine they feed).
function fp_dft5_consts(c::UInt64, F::FpCtx)
    inv2 = Float64(invmod(UInt64(2), fp_prime(F)))
    c1 = fp_reduce(Float64(c), F)
    c2 = fp_mulmod(c1, c1, F); c3 = fp_mulmod(c2, c1, F); c4 = fp_mulmod(c3, c1, F)
    inv4 = fp_mulmod(inv2, inv2, F)
    return Float64[fp_minbal(-inv4, F),
                   fp_minbal(fp_mulmod((c1 + c4) - (c2 + c3), inv4, F), F),
                   fp_minbal(fp_mulmod(c1 - c4, inv2, F), F),
                   fp_minbal(fp_mulmod(c2 - c3, inv2, F), F)]
end

# Winograd constants for the 7-point DFT (see fp_dft7).  Pairing x_j with
# x_{7-j} splits the 6x6 DFT matrix into half-sum/half-difference quadrants
# A_{jr} = (c^{jr} + c^{-jr})/2, B_{jr} = (c^{jr} - c^{-jr})/2; reindexing the
# pairs by the generator 3 mod 7 (pair orbit 1 -> 3 -> 2, and 3³ ≡ -1 folds
# onto the same pairs) makes the A quadrant a 3-point cyclic correlation with
# a = (A_1, A_3, A_2) and the B quadrant a skew-cyclic one with
# b = (B_1, B_3, B_2).  Each 3-convolution is done CRT-style in 4 mulmods
# (x³∓1 = (x∓1)(x²±x+1)), with the 1/3 of the reconstruction idempotents
# folded into the constants:
#   [1] K1  = (a0+a1+a2)/3      [5] Kn  = (b0-b1+b2)/3
#   [2] b'0 = (a0-a2)/3         [6] n0  = (b0-b2)/3
#   [3] b'1 = (a1-a2)/3         [7] n1  = (b1+b2)/3
#   [4] b'0+b'1                 [8] n0+n1
function fp_dft7_consts(c::UInt64, F::FpCtx)
    p = fp_prime(F)
    inv2 = Float64(invmod(UInt64(2), p))
    inv3 = Float64(invmod(UInt64(3), p))
    cj = [Float64(powermod(c, j, p)) for j in 1:6]
    A(j) = fp_mulmod(cj[j] + cj[7-j], inv2, F)
    B(j) = fp_mulmod(cj[j] - cj[7-j], inv2, F)
    a0, a1, a2 = A(1), A(3), A(2)
    b0, b1, b2 = B(1), B(3), B(2)
    bp0 = fp_minbal(fp_mulmod(a0 - a2, inv3, F), F)
    bp1 = fp_minbal(fp_mulmod(a1 - a2, inv3, F), F)
    n0 = fp_minbal(fp_mulmod(b0 - b2, inv3, F), F)
    n1 = fp_minbal(fp_mulmod(b1 + b2, inv3, F), F)
    return Float64[fp_minbal(fp_mulmod((a0 + a1) + a2, inv3, F), F), bp0, bp1,
                   fp_minbal(bp0 + bp1, F),
                   fp_minbal(fp_mulmod((b0 + b2) - b1, inv3, F), F), n0, n1,
                   fp_minbal(n0 + n1, F)]
end

# Minimal balanced (|w| <= p/2) Float64 powers r^0, r^1, ..., r^(n-1) mod p,
# generated 8 lanes at a time; the per-step fp_minbal keeps the chain minimal
# (raw mulmod output drifts to ~0.57p).
function fp_twiddles(r::UInt64, n::Int, F::FpCtx{P}) where P
    np = cld(n, 8) << 3
    t = Vector{Float64}(undef, np)
    r8 = fp_reduce(Float64(powermod(r, 8, P)), F)
    r8p = r8 / P
    v = fp_reduce(VF8(ntuple(k -> Float64(powermod(r, k - 1, P)), 8)), F)
    @inbounds for j in 1:8:np
        v = fp_minbal(v, F)
        SIMD.vstore(v, t, j)
        v = fp_mulmod(v, r8, r8p, F)
    end
    resize!(t, n)
    return t
end

# One transform pass: rot holds the DFT combine constants, tw the m-1
# per-residue twiddle tables (all minimal balanced), and span is the block
# size the stage processes at each offset.
struct FpNttStage
    m::Int
    span::Int
    Q::Int
    rot::Vector{Float64}
    tw::Vector{Vector{Float64}}
end

function build_fp_4(table::Vector{Float64}, N2::Int, L::Int, wi::Float64)
    q = L >> 2
    st = N2 ÷ L
    len = max(q, 8)     # q < 8 stages replicate the pattern so fp_smallq! can vload it
    tw = [Vector{Float64}(undef, len) for _ in 1:3]
    @inbounds for j in 0:len-1
        jm = j % q
        tw[1][j+1] = table[jm*st+1]
        tw[2][j+1] = table[2jm*st+1]
        tw[3][j+1] = table[3jm*st+1]
    end
    return FpNttStage(4, N2, q, [wi], tw)
end

function build_fp_odd(m::Int, span::Int, root::UInt64, F::FpCtx)
    Q = span ÷ m
    c = powermod(root, Q, fp_prime(F))
    rot = m == 5 ? fp_dft5_consts(c, F) :
          m == 7 ? fp_dft7_consts(c, F) :
          Float64[fp_reduce(Float64(powermod(c, r, fp_prime(F))), F) for r in 1:m-1]
    tw = Vector{Vector{Float64}}(undef, m - 1)
    for r in 1:m-1
        ωr = powermod(root, r, fp_prime(F))
        tw[r] = fp_twiddles(ωr, Q, F)
    end
    return FpNttStage(m, span, Q, rot, tw)
end

struct FpNttPlan{C<:FpCtx}
    ctx::C
    N::Int
    ninv::Float64   # 1/N mod p, minimal balanced; applied in the unpack
    ninvp::Float64  # ninv/p, its precomputed fp_mulmod quotient companion
    stages::Vector{FpNttStage}  # forward order: odd radices, then the radix-4 tower
end

function build_fp_plan(N::Int, F::FpCtx)
    @assert (fp_prime(F) - 1) % N == 0   # so ω below is a primitive Nth root
    ω = powermod(F.gen, (fp_prime(F) - 1) ÷ N, fp_prime(F))

    stages = FpNttStage[]
    span = N
    r = ω
    # peel the arithmetic-heaviest radix first. Earlier stages are more memory bound
    for mf in (7, 5, 3, 3)
        if span % mf == 0
            push!(stages, build_fp_odd(mf, span, r, F))
            r = powermod(r, mf, fp_prime(F))
            span ÷= mf
        end
    end
    # N = m·2^k: the peel above strips odd factors so N must be 2^k.
    # 4 <= 2^k | P-1 required for NTT to be valid.
    @assert ispow2(span) && 2 <= trailing_zeros(span) <= two_adicity(F)

    tw = fp_twiddles(r, span, F)
    wi = fp_reduce(Float64(powermod(r, span >> 2, fp_prime(F))), F)
    L = span
    while L >= 4
        push!(stages, build_fp_4(tw, span, L, wi))
        L >>= 2
    end
    # leftover radix-2 pass for odd k
    isodd(trailing_zeros(span)) &&
        push!(stages, FpNttStage(2, N, N >> 1, Float64[], Vector{Float64}[]))
    ninv = fp_reduce(Float64(invmod(N % UInt64, fp_prime(F))), F)
    return FpNttPlan(F, N, ninv, ninv / fp_prime(F), stages)
end

mutable struct FpNttPlanCache
    @atomic plans::Dict{Tuple{Int,UInt64},FpNttPlan}
end
const FPNTT_PLAN_LOCK = ReentrantLock()
const FPNTT_PLAN_CACHE = FpNttPlanCache(Dict{Tuple{Int,UInt64},FpNttPlan}())
function fp_ntt_plan(N::Int, F::C) where {C<:FpCtx}
    key = (N, fp_prime(F))
    plan = get(@atomic(:acquire, FPNTT_PLAN_CACHE.plans), key, nothing)
    plan === nothing || return plan::FpNttPlan{C}
    r = lock(FPNTT_PLAN_LOCK) do
        plans = @atomic :acquire FPNTT_PLAN_CACHE.plans
        cached = get(plans, key, nothing)
        cached === nothing || return cached
        built = build_fp_plan(N, F)
        next = copy(plans)
        next[key] = built
        @atomic :release FPNTT_PLAN_CACHE.plans = next
        return built
    end
    return r::FpNttPlan{C}
end

# ---------------------------------------------------------------------------
# Odd-radix stages: Winograd radix-3 and radix-5, symmetric-pair radix-7.
# A stage is twiddle+reduce with the combine before it (forward) or after it
# (reverse); all three share that twiddle+reduce structure.

# Raw Winograd 3-point combine (u = ω3·(b - c) folds the matrix into one
# mulmod). Every output, including ω^0, comes back unreduced: the forward
# reduces it in the shared block; the reverse stores it raw (<= ~2.4p),
# legal because radix-3 runs last in the reverse, so only the unpack's
# 4p-domain 1/N mulmod loads it.
@inline function fp_dft3(a, b, c, ω3, F::FpCtx)
    u = fp_mulmod(b - c, ω3, F)
    return a + (b + c), (a - c) + u, (a - b) - u
end

function fp_radix3!(x::AbstractVector{Float64}, o::Int, st::FpNttStage, F::FpCtx,
                    ::Val{FWD}) where {FWD}
    (;Q, tw, rot) = st
    ω3 = rot[1]
    tw1, tw2 = tw[1], tw[2]
    @inbounds for j in 0:8:Q-8
        i0 = o + j + 1
        y0 = SIMD.vload(VF8, x, i0)
        y1 = SIMD.vload(VF8, x, i0 + Q)
        y2 = SIMD.vload(VF8, x, i0 + 2Q)
        if FWD
            y0, y1, y2 = fp_dft3(y0, y1, y2, ω3, F)
        end
        y0 = fp_reduce(y0, F)
        y1 = fp_mulmod(y1, SIMD.vload(VF8, tw1, j + 1), F)
        y2 = fp_mulmod(y2, SIMD.vload(VF8, tw2, j + 1), F)
        if !FWD
            y0, y1, y2 = fp_dft3(y0, y1, y2, ω3, F)
        end
        SIMD.vstore(y0, x, i0)
        SIMD.vstore(y1, x, i0 + Q)
        SIMD.vstore(y2, x, i0 + 2Q)
    end
    @inbounds for j in (Q & ~7):Q-1
        i0 = o + j + 1
        y0 = x[i0]
        y1 = x[i0+Q]
        y2 = x[i0+2Q]
        if FWD
            y0, y1, y2 = fp_dft3(y0, y1, y2, ω3, F)
        end
        y0 = fp_reduce(y0, F)
        y1 = fp_mulmod(y1, tw1[j+1], F)
        y2 = fp_mulmod(y2, tw2[j+1], F)
        if !FWD
            y0, y1, y2 = fp_dft3(y0, y1, y2, ω3, F)
        end
        x[i0] = y0
        x[i0+Q] = y1
        x[i0+2Q] = y2
    end
    return x
end

# Winograd order-5 DFT combine, 6 mulmods by ±-pairing the outputs:
# y1+y4 = 2·x0 + a·(x1+x4) + b·(x2+x3) for
# a = ω + ω^4, b = ω^2 + ω^3, and a + b = -1 rewrites that as
# 2·x0 - S/2 + ((a-b)/2)·D — one mulmod, not two.
# With the halving folded into k1..k4 (fp_dft5_consts),
# r, q, g1, g2 below are the half-sums and half-differences of the pairs:
#   (y1+y4)/2 = x0 + k1·S + k2·D = r + q     (y1-y4)/2 = k3·t3 + k4·t4 = g1
#   (y2+y3)/2 = r - q                        (y2-y3)/2 = k4·t3 - k3·t4 = g2
# so y1, y4 = (r+q) ± g1 and y2, y3 = (r-q) ± g2.
# Bounds:
# for inputs |xi| <= ~1.0p, S and D stay <= ~3.6p (round argument
# S·k <= 1.8p with minimal constants); reducing x0 + k1·S closes the chain,
# leaving every output <= ~2.9p — inside the mulmod domain of whatever loads
# it next. y0 comes back unreduced; callers reduce it in both directions
# (the reverse column sum reaches ~4.1p raw, past comfortable margins).
@inline function fp_dft5(x0, x1, x2, x3, x4, k1, k2, k3, k4, F::FpCtx)
    t1 = x1 + x4; t3 = x1 - x4
    t2 = x2 + x3; t4 = x2 - x3
    S = t1 + t2;  D = t1 - t2
    y0 = x0 + S
    r = fp_reduce(x0 + fp_mulmod(S, k1, F), F)
    q = fp_mulmod(D, k2, F)
    g1 = fp_mulmod(t3, k3, F) + fp_mulmod(t4, k4, F)
    g2 = fp_mulmod(t3, k4, F) - fp_mulmod(t4, k3, F)
    u = r + q; v = r - q
    return y0, u + g1, v + g2, v - g2, u - g1
end

function fp_radix5!(x::AbstractVector{Float64}, o::Int, st::FpNttStage, F::FpCtx,
                    ::Val{FWD}) where {FWD}
    (;Q, tw, rot) = st
    k1, k2, k3, k4 = rot
    tw1, tw2, tw3, tw4 = tw[1], tw[2], tw[3], tw[4]
    @inbounds for j in 0:8:Q-8
        i0 = o + j + 1
        y0 = SIMD.vload(VF8, x, i0)
        y1 = SIMD.vload(VF8, x, i0 + Q)
        y2 = SIMD.vload(VF8, x, i0 + 2Q)
        y3 = SIMD.vload(VF8, x, i0 + 3Q)
        y4 = SIMD.vload(VF8, x, i0 + 4Q)
        if FWD
            y0, y1, y2, y3, y4 = fp_dft5(y0, y1, y2, y3, y4, k1, k2, k3, k4, F)
        end
        y0 = fp_reduce(y0, F)
        y1 = fp_mulmod(y1, SIMD.vload(VF8, tw1, j + 1), F)
        y2 = fp_mulmod(y2, SIMD.vload(VF8, tw2, j + 1), F)
        y3 = fp_mulmod(y3, SIMD.vload(VF8, tw3, j + 1), F)
        y4 = fp_mulmod(y4, SIMD.vload(VF8, tw4, j + 1), F)
        if !FWD
            y0, y1, y2, y3, y4 = fp_dft5(y0, y1, y2, y3, y4, k1, k2, k3, k4, F)
            y0 = fp_reduce(y0, F)
        end
        SIMD.vstore(y0, x, i0)
        SIMD.vstore(y1, x, i0 + Q)
        SIMD.vstore(y2, x, i0 + 2Q)
        SIMD.vstore(y3, x, i0 + 3Q)
        SIMD.vstore(y4, x, i0 + 4Q)
    end
    @inbounds for j in (Q & ~7):Q-1
        i0 = o + j + 1
        y0 = x[i0]
        y1 = x[i0+Q]
        y2 = x[i0+2Q]
        y3 = x[i0+3Q]
        y4 = x[i0+4Q]
        if FWD
            y0, y1, y2, y3, y4 = fp_dft5(y0, y1, y2, y3, y4, k1, k2, k3, k4, F)
        end
        y0 = fp_reduce(y0, F)
        y1 = fp_mulmod(y1, tw1[j+1], F)
        y2 = fp_mulmod(y2, tw2[j+1], F)
        y3 = fp_mulmod(y3, tw3[j+1], F)
        y4 = fp_mulmod(y4, tw4[j+1], F)
        if !FWD
            y0, y1, y2, y3, y4 = fp_dft5(y0, y1, y2, y3, y4, k1, k2, k3, k4, F)
            y0 = fp_reduce(y0, F)
        end
        x[i0] = y0
        x[i0+Q] = y1
        x[i0+2Q] = y2
        x[i0+3Q] = y3
        x[i0+4Q] = y4
    end
    return x
end

# Winograd 7-point DFT (constants and derivation in fp_dft7_consts): with
# s_j = x_j + x_{7-j}, d_j = x_j - x_{7-j} (j = 1..3),
#   y_r     = x0 + u_r + v_r,   y_{7-r} = x0 + u_r - v_r
# where (u_1,u_3,u_2) is the cyclic 3-convolution of (s_1,s_2,s_3) with the
# A-constants and (v_1,v_3,v_2) the skew-cyclic one of the d's with the
# B-constants — 8 mulmods per column instead of the DFT matrix's 36.
# Bounds: RED (reverse only) closes the twiddled s,d at <= ~0.55p, so every
# convolution input sum is <= ~2.2p (round argument <= ~1.1p) and mulmod
# outs are <= ~0.6p; the reconstruction sums reach <= ~5p < 2^53/p before
# the u/v reduces, so every y is <= |x0| + ~1.1p.  In the forward, radix-7
# runs first and its inputs are pack chunks << p, so the entry reduces are
# skipped but the u/v reduces stay (mulmod outputs are ~p/2 regardless of
# input size, and 5 of them accumulate).  y0 comes back unreduced; the
# forward's shared block reduces it, the reverse re-reduces after the
# combine.
@inline function fp_dft7(x0, x1, x2, x3, x4, x5, x6, rot::NTuple{8},
                         F::FpCtx, ::Val{RED}) where {RED}
    s1 = x1 + x6; d1 = x1 - x6
    s2 = x2 + x5; d2 = x2 - x5
    s3 = x3 + x4; d3 = x3 - x4
    if RED
        s1 = fp_reduce(s1, F); s2 = fp_reduce(s2, F); s3 = fp_reduce(s3, F)
        d1 = fp_reduce(d1, F); d2 = fp_reduce(d2, F); d3 = fp_reduce(d3, F)
    end
    S = (s1 + s2) + s3
    y0 = x0 + S
    m1 = fp_mulmod(S, rot[1], F)
    va = s1 - s3; vb = s2 - s3
    w0 = fp_mulmod(va, rot[2], F)
    w1 = fp_mulmod(vb, rot[3], F)
    w2 = fp_mulmod(va + vb, rot[4], F)
    mn = fp_mulmod((d1 + d2) - d3, rot[5], F)
    wa = d1 + d3; wb = -(d2 + d3)
    q0 = fp_mulmod(wa, rot[6], F)
    q1 = fp_mulmod(wb, rot[7], F)
    q2 = fp_mulmod(wa + wb, rot[8], F)
    u1 = fp_reduce(m1 + (3w0 - w2), F)
    u3 = fp_reduce((m1 + 2w2) - 3(w0 + w1), F)
    u2 = fp_reduce(m1 + (3w1 - w2), F)
    r0 = q0 - q1; r1 = q2 - q0
    v1 = fp_reduce(mn + (2r0 + r1), F)
    v3 = fp_reduce((r0 + 2r1) - mn, F)
    v2 = fp_reduce((mn - r0) + r1, F)
    a1 = x0 + u1; a2 = x0 + u2; a3 = x0 + u3
    return y0, a1 + v1, a2 + v2, a3 + v3, a3 - v3, a2 - v2, a1 - v1
end

function fp_radix7!(x::AbstractVector{Float64}, o::Int, st::FpNttStage, F::FpCtx,
                    ::Val{FWD}) where {FWD}
    (;Q, tw, rot) = st
    # hoisted: loop-invariant, but LLVM must assume the x stores alias rot
    tws = ntuple(t -> tw[t], Val(6))
    rs = ntuple(t -> (@inbounds rot[t]), Val(8))
    rv = map(VF8, rs)
    @inbounds for j in 0:8:Q-8
        i0 = o + j + 1
        y0 = SIMD.vload(VF8, x, i0)
        Base.Cartesian.@nexprs 6 t -> y_t = SIMD.vload(VF8, x, i0 + t * Q)
        if FWD
            y0, y_1, y_2, y_3, y_4, y_5, y_6 =
                fp_dft7(y0, y_1, y_2, y_3, y_4, y_5, y_6, rv, F, Val(false))
        end
        y0 = fp_reduce(y0, F)
        Base.Cartesian.@nexprs 6 t -> y_t = fp_mulmod(y_t, SIMD.vload(VF8, tws[t], j + 1), F)
        if !FWD
            y0, y_1, y_2, y_3, y_4, y_5, y_6 =
                fp_dft7(y0, y_1, y_2, y_3, y_4, y_5, y_6, rv, F, Val(true))
            y0 = fp_reduce(y0, F)
        end
        SIMD.vstore(y0, x, i0)
        Base.Cartesian.@nexprs 6 t -> SIMD.vstore(y_t, x, i0 + t * Q)
    end
    @inbounds for j in (Q & ~7):Q-1
        i0 = o + j + 1
        y0 = x[i0]
        Base.Cartesian.@nexprs 6 t -> y_t = x[i0+t*Q]
        if FWD
            y0, y_1, y_2, y_3, y_4, y_5, y_6 =
                fp_dft7(y0, y_1, y_2, y_3, y_4, y_5, y_6, rs, F, Val(false))
        end
        y0 = fp_reduce(y0, F)
        Base.Cartesian.@nexprs 6 t -> y_t = fp_mulmod(y_t, tws[t][j+1], F)
        if !FWD
            y0, y_1, y_2, y_3, y_4, y_5, y_6 =
                fp_dft7(y0, y_1, y_2, y_3, y_4, y_5, y_6, rs, F, Val(true))
            y0 = fp_reduce(y0, F)
        end
        x[i0] = y0
        Base.Cartesian.@nexprs 6 t -> x[i0+t*Q] = y_t
    end
    return x
end

# ---------------------------------------------------------------------------
# Radix-4 butterfly core, shared by both directions (the DFT4 matrix is
# symmetric).  y0 comes back unreduced: the forward reduces it before
# storing; the reverse stores it raw.
@inline function fp_dft4(a, b, c, d, wi, F::FpCtx)
    apc = a + c
    amc = a - c
    bpd = b + d
    ibmd = fp_mulmod(b - d, wi, F)
    return apc + bpd, amc + ibmd, apc - bpd, amc - ibmd
end

# Small-q pow2 stages (q in (1,2,4)): butterfly partners sit closer together
# than a vector, so 32 consecutive elements (four V8 loads) are shuffled
# into quarter-role vectors a,b,c,d, put through the ordinary vector
# butterfly, and shuffled back.  The twiddle tables' repeated 8-entry
# patterns line up with the gathered lane order.
#
# Lane l of role t lives at global position (l÷Q)·4Q + t·Q + l%Q within the
# 32 elements; scatter applies the inverse permutation.  The generators emit one
# index-tuple shuffle per output vector, which LLVM folds into the same machine
# shuffles as hand-written per-Q versions.
const IOTA16 = ntuple(i -> i - 1, 16)

@generated function ntt_gather4(::Val{Q}, v0, v1, v2, v3) where {Q}
    role(t) = ntuple(l -> ((l - 1) ÷ Q) * 4Q + t * Q + (l - 1) % Q, 8)
    return quote
        $(Expr(:meta, :inline))
        lo = SIMD.shufflevector(v0, v1, Val(IOTA16))
        hi = SIMD.shufflevector(v2, v3, Val(IOTA16))
        return (SIMD.shufflevector(lo, hi, Val($(role(0)))),
                SIMD.shufflevector(lo, hi, Val($(role(1)))),
                SIMD.shufflevector(lo, hi, Val($(role(2)))),
                SIMD.shufflevector(lo, hi, Val($(role(3)))))
    end
end

@generated function ntt_scatter4(::Val{Q}, a, b, c, d) where {Q}
    # global position g pulls role (g%4Q)÷Q, lane (g÷4Q)·Q + g%Q; roles are
    # concatenated in order, so the combined source index is role·8 + lane
    out(o) = ntuple(8) do l
        g = 8o + l - 1
        ((g % 4Q) ÷ Q) * 8 + (g ÷ 4Q) * Q + g % Q
    end
    return quote
        $(Expr(:meta, :inline))
        ab = SIMD.shufflevector(a, b, Val(IOTA16))
        cd = SIMD.shufflevector(c, d, Val(IOTA16))
        return (SIMD.shufflevector(ab, cd, Val($(out(0)))),
                SIMD.shufflevector(ab, cd, Val($(out(1)))),
                SIMD.shufflevector(ab, cd, Val($(out(2)))),
                SIMD.shufflevector(ab, cd, Val($(out(3)))))
    end
end

function fp_smallq!(x::AbstractVector{Float64}, o::Int, ::Val{Q},
                    stg::FpNttStage, F::FpCtx, ::Val{FWD}) where {Q,FWD}
    wi = @inbounds stg.rot[1]
    vw1 = SIMD.vload(VF8, stg.tw[1], 1)
    vw2 = SIMD.vload(VF8, stg.tw[2], 1)
    vw3 = SIMD.vload(VF8, stg.tw[3], 1)
    @inbounds for i0 in o+1:32:o+stg.span-31
        v0 = SIMD.vload(VF8, x, i0)
        v1 = SIMD.vload(VF8, x, i0 + 8)
        v2 = SIMD.vload(VF8, x, i0 + 16)
        v3 = SIMD.vload(VF8, x, i0 + 24)
        y0, y1, y2, y3 = ntt_gather4(Val(Q), v0, v1, v2, v3)
        if FWD
            y0, y1, y2, y3 = fp_dft4(y0, y1, y2, y3, wi, F)
        end
        y0 = fp_reduce(y0, F)
        y1 = fp_mulmod(y1, vw1, F)
        y2 = fp_mulmod(y2, vw2, F)
        y3 = fp_mulmod(y3, vw3, F)
        if !FWD
            y0, y1, y2, y3 = fp_dft4(y0, y1, y2, y3, wi, F)
        end
        o0, o1, o2, o3 = ntt_scatter4(Val(Q), y0, y1, y2, y3)
        SIMD.vstore(o0, x, i0)
        SIMD.vstore(o1, x, i0 + 8)
        SIMD.vstore(o2, x, i0 + 16)
        SIMD.vstore(o3, x, i0 + 24)
    end
    return x
end

# Leftover radix-2 stage for odd k, twiddle-free and shared by both
# directions (the DFT2 matrix is its own transpose), so it ignores FWD.
# Outputs are reduced: raw they reach ~2.4p, and the forward feeds the pointwise
# stage, whose product of two values must stay within the 2^51·p mulmod domain.
# The only stage relying on LLVM's auto-vectorizer, hence s based at 0 with o
# folded into the index: an offset-based induction variable loses every one of
# the 18 <8 x double> ops (checked in the IR).
function fp_radix2!(x::AbstractVector{Float64}, o::Int, st::FpNttStage, F::FpCtx, ::Val)
    @inbounds for s in 0:2:st.span-1
        u = x[o+s+1]
        v = x[o+s+2]
        x[o+s+1] = fp_reduce(u + v, F)
        x[o+s+2] = fp_reduce(u - v, F)
    end
    return x
end

# One radix-4 stage (butterfly span 4Q) over the length st.span block at offset o.
# FWD only picks which side of the shared twiddle+reduce pass the DFT4 combine
# runs on.  Reverse outputs store raw: the next stage's ω^0 reduce or the
# unpack's 1/N mulmod absorbs them.
function fp_radix4!(x::AbstractVector{Float64}, o::Int, st::FpNttStage, F::FpCtx,
                    ::Val{FWD}) where {FWD}
    (;Q, tw, rot) = st
    wi = @inbounds rot[1]
    if Q >= 8
        tw1, tw2, tw3 = tw[1], tw[2], tw[3]
        for s in o:4Q:o+st.span-1
            @inbounds for j in 0:8:Q-8
                i0 = s + j + 1
                y0 = SIMD.vload(VF8, x, i0)
                y1 = SIMD.vload(VF8, x, i0 + Q)
                y2 = SIMD.vload(VF8, x, i0 + 2Q)
                y3 = SIMD.vload(VF8, x, i0 + 3Q)
                if FWD
                    y0, y1, y2, y3 = fp_dft4(y0, y1, y2, y3, wi, F)
                end
                y0 = fp_reduce(y0, F)
                y1 = fp_mulmod(y1, SIMD.vload(VF8, tw1, j + 1), F)
                y2 = fp_mulmod(y2, SIMD.vload(VF8, tw2, j + 1), F)
                y3 = fp_mulmod(y3, SIMD.vload(VF8, tw3, j + 1), F)
                if !FWD
                    y0, y1, y2, y3 = fp_dft4(y0, y1, y2, y3, wi, F)
                end
                SIMD.vstore(y0, x, i0)
                SIMD.vstore(y1, x, i0 + Q)
                SIMD.vstore(y2, x, i0 + 2Q)
                SIMD.vstore(y3, x, i0 + 3Q)
            end
        end
    elseif st.span >= 32
        Q == 4 ? fp_smallq!(x, o, Val(4), st, F, Val(FWD)) :
        Q == 2 ? fp_smallq!(x, o, Val(2), st, F, Val(FWD)) :
                 fp_smallq!(x, o, Val(1), st, F, Val(FWD))
    else
        tw1, tw2, tw3 = tw[1], tw[2], tw[3]
        for s in o:4Q:o+st.span-1
            @inbounds for j in 0:Q-1
                y0 = x[s+j+1]
                y1 = x[s+j+Q+1]
                y2 = x[s+j+2Q+1]
                y3 = x[s+j+3Q+1]
                if FWD
                    y0, y1, y2, y3 = fp_dft4(y0, y1, y2, y3, wi, F)
                end
                y0 = fp_reduce(y0, F)
                y1 = fp_mulmod(y1, tw1[j+1], F)
                y2 = fp_mulmod(y2, tw2[j+1], F)
                y3 = fp_mulmod(y3, tw3[j+1], F)
                if !FWD
                    y0, y1, y2, y3 = fp_dft4(y0, y1, y2, y3, wi, F)
                end
                x[s+j+1] = y0
                x[s+j+Q+1] = y1
                x[s+j+2Q+1] = y2
                x[s+j+3Q+1] = y3
            end
        end
    end
    return x
end

@inline function fp_stage!(x::AbstractVector{Float64}, st::FpNttStage, plan::FpNttPlan,
                           fwd::Val)
    F = plan.ctx
    for o in 0:st.span:plan.N-1
        st.m == 2 ? fp_radix2!(x, o, st, F, fwd) :
        st.m == 3 ? fp_radix3!(x, o, st, F, fwd) :
        st.m == 4 ? fp_radix4!(x, o, st, F, fwd) :
        st.m == 5 ? fp_radix5!(x, o, st, F, fwd) :
                    fp_radix7!(x, o, st, F, fwd)
    end
    return x
end

function fp_ntt_fwd!(x::AbstractVector{Float64}, plan::FpNttPlan)
    for stage in plan.stages
        fp_stage!(x, stage, plan, Val(true))
    end
    return x
end

# Transpose of fp_ntt_fwd!: same twiddles, stages reversed.
function fp_ntt_rev!(x::AbstractVector{Float64}, plan::FpNttPlan)
    for stage in Iterators.reverse(plan.stages)
        fp_stage!(x, stage, plan, Val(false))
    end
    return x
end

# ---------------------------------------------------------------------------
# Coefficient-domain layer, shared by both pipelines: size the transform
# (ntt_len), split limbs into chunk coefficients (fp_ntt_pack!), and multiply
# lanewise between the transforms (fp_ntt_pointwise!).

# b-bit chunk extraction into Float64 points; chunks < 2^b <= 2^52 are exact
# doubles and, being smaller than every prime, already canonical residues.
# Overwrites all of x (chunks, then zero fill), so callers can recycle a
# spent transform buffer.
function fp_ntt_pack!(x::AbstractVector{Float64}, limbs::Memory{Limb}, lo::Int, n::Int,
                      b::Int, nch::Int)
    N = length(x)
    mask = (UInt64(1) << b) - 1
    imax = min(nch - 1, fld(64 * (n - 1) - 1, b))
    @inbounds for i in 0:imax
        bit = i * b
        w = bit >> 6
        sh = bit & 63
        c = (limbs[lo+w+1] >>> sh) | (limbs[lo+w+2] << (64 - sh))
        x[i+1] = c & mask
    end
    @inbounds for i in imax+1:nch-1
        bit = i * b
        w = bit >> 6
        sh = bit & 63
        c = limbs[lo+w+1] >>> sh
        if sh + b > 64 && w + 2 <= n
            c |= limbs[lo+w+2] << (64 - sh)
        end
        x[i+1] = c & mask
    end
    fill!(view(x, nch+1:N), 0.0)
    return x
end

function fp_ntt_pointwise!(xa::AbstractVector{Float64}, xb::AbstractVector{Float64}, F::FpCtx)
    n = length(xa)
    @inbounds for i in 1:8:n-7
        SIMD.vstore(fp_mulmod(SIMD.vload(VF8, xa, i), SIMD.vload(VF8, xb, i), F), xa, i)
    end
    @inbounds for i in (n & ~7)+1:n
        xa[i] = fp_mulmod(xa[i], xb[i], F)
    end
    return xa
end

# Garner recombination + limb streaming.  (Unpack loads reverse-transform
# values up to ~3.3p raw; ninv is minimal balanced so the 1/N round argument
# stays <= ~1.7p < 2^51.)  fp_ntt_rev! hands back N·c index-
# reversed (coefficient i at (N-i) mod N); this loop scales each residue by 1/N
# and Garner-combines the two mod-p residues of every base-2^b coefficient:
#   c = c1 + p1·u,  u = (c2 - c1)·p1^-1 mod p2.
# c is never materialized — by linearity Σ cᵢ·2^(b·i) = S1 + p1·S2 with
# S1 = Σ c1ᵢ·2^(b·i) and S2 = Σ uᵢ·2^(b·i), so the two sums stream independently:
# S1 into r, and S2 through a fixed 32-limb window that addmul_1! folds into r at
# kernel speed as it fills, while those limbs are still hot (see fp_fold_s2!).
#
# Each c1, u is pre-split into window halves lo = v << s, hi = v >> (64 - s) so
# the scalar accumulate is pure add-with-carry, with s = (b·i) mod 64 data-
# independent (the flush drops s by exactly 64).  The 128-bit window never
# overflows: values are < 2^50, s < 64, and each flushed limb is final.  Note c1
# must be canonicalized before fp_reduce(v1, F2), since v1 - p1 is a different
# value mod p2.

# Unscale a raw value pair by 1/N and Garner-combine (scalar or VF8).
@inline function fp_garner(w1, w2, n1, n1p, n2, n2p)
    v1 = fp_canonical(fp_mulmod(w1, n1, n1p, FP_CTX1), FP_CTX1)
    v2 = fp_mulmod(w2, n2, n2p, FP_CTX2)
    u = fp_mulmod(v2 - fp_reduce(v1, FP_CTX2), FP_GARNER_G, FP_GARNER_GP, FP_CTX2)
    return v1, fp_canonical(u, FP_CTX2)
end

# One scalar coefficient, as canonical UInt64s.
@inline function fp_unpack_coeff(x1::AbstractVector{Float64}, x2::AbstractVector{Float64}, idx::Int,
                                 n1::Float64, n1p::Float64, n2::Float64, n2p::Float64)
    v1, u = fp_garner(x1[idx], x2[idx], n1, n1p, n2, n2p)
    return unsafe_trunc(UInt64, v1), unsafe_trunc(UInt64, u)
end

# The < 8-wide head/tail fallback for the vector loop's coefficient packing:
# cnt (<= 8) coefficients from i into the stage buffer (quarters lo1|hi1|lo2|hi2),
# lane j shifted by (s + j·b) & 63.  Coefficient 0 wraps to x1[1]; the rest read
# descending x1[N-·+1].  @noinline keeps its fp_mulmod chains out of
# fp_ntt_unpack2!'s hot loop, whose adc recognition degrades when it is bloated.
@noinline function fp_unpack_scalar!(stage::Vector{UInt64}, x1::AbstractVector{Float64},
                                 x2::AbstractVector{Float64}, N::Int, i::Int, cnt::Int,
                                 s::Int, b::Int, n1::Float64, n1p::Float64,
                                 n2::Float64, n2p::Float64)
    @inbounds for j in 0:cnt-1
        c = i + j
        idx = c == 0 ? 1 : N - c + 1
        c1, uu = fp_unpack_coeff(x1, x2, idx, n1, n1p, n2, n2p)
        sh = (s + j * b) & 63
        stage[j+1]  = c1 << sh
        stage[j+9]  = (c1 >> 1) >> (63 - sh) # same as >> (64 - sh), avoiding shift by 64
        stage[j+17] = uu << sh
        stage[j+25] = (uu >> 1) >> (63 - sh)
    end
    return stage
end

# Fold cnt (<= 8) pre-shifted coefficients from the word buffer into both
# 128-bit windows, flushing a finished limb whenever bitpos crosses a limb
# boundary.  bitpos is the absolute bit offset of the next coefficient (i·b),
# so the windows cover bits [64·(bitpos>>6), +128).  Spelled out
# inline rather than via a per-coefficient helper on purpose: a per-lane
# function boundary drops LLVM's adc carry chain (~60% slower).  At the hot call
# site cnt = 8 is a literal, so this unrolls to the full 16-adc chain.
@inline function fp_accum_words!(r::Memory{Limb}, ro::Int, s2::Memory{Limb}, nw::Int,
                                w1::UInt128, w2::UInt128, bitpos::Int, b::Int,
                                words::Vector{UInt64}, cnt::Int)
    s = bitpos & 63
    outw = (bitpos >> 6) + 1
    @inbounds for j in 1:cnt
        w1 += (UInt128(words[j+8]) << 64) | words[j]
        w2 += (UInt128(words[j+24]) << 64) | words[j+16]
        s += b
        if s >= 64
            r[ro+outw] = w1 % UInt64
            nw += 1
            s2[nw] = w2 % UInt64
            w1 >>= 64
            w2 >>= 64
            outw += 1
            s -= 64
        end
    end
    return w1, w2, nw
end

# Fold the pending stream-2 limbs into r as p1·S2, incrementally: they sit at
# r limbs fold+1 .. fold+nw, all already stored, so addmul_1!'s carry-out lands
# on limb fold+nw+1 — which is exactly w1's low limb, still in flight.  Adding
# it to r instead would be wrong: that limb has not been stored yet, and the
# store is a store, not an add.  Called once the window holds >= 8 limbs so
# addmul_1! runs its SIMD block; the final call drains whatever is left, whose
# carry must be zero since the product fits in rn limbs.
@inline function fp_fold_s2!(r::Memory{Limb}, ro::Int, s2::Memory{Limb}, nw::Int,
                            fold::Int, w1::UInt128)
    nw == 0 && return w1, fold
    w1 += addmul_1!(r, ro + fold, s2, 0, nw, fp_prime(FP_CTX1))
    return w1, fold + nw
end

function fp_ntt_unpack2!(r::Memory{Limb}, ro::Int, rn::Int,
                         x1::AbstractVector{Float64}, x2::AbstractVector{Float64}, nconv::Int,
                         b::Int, n1::Float64, n1p::Float64, n2::Float64, n2p::Float64)
    N = length(x1)
    s2 = Memory{Limb}(undef, 32)
    nw = 0     # limbs pending in s2
    fold = 0   # r limbs already folded
    # Pre-shifted coefficient words awaiting the carry accumulate, quarters
    # lo1 | hi1 | lo2 | hi2.  fp_accum_words! reads lanes at a runtime index;
    # Vec lane extraction with a non-constant index spills to the stack
    # (~30% slower than this explicit round-trip).
    wordbuf = Vector{UInt64}(undef, 32)
    ub = UInt64(b)
    # canonical values are < p < 2^50, so v + 1.5·2^52 pins the exponent and
    # leaves v in the low 51 mantissa bits: int(v) is a reinterpret-and-mask
    mask = UInt64(2)^51 - 1
    REV8 = Val((7, 6, 5, 4, 3, 2, 1, 0))
    vlane = V8(ntuple(k -> UInt64(k - 1) * ub, 8))
    w1 = UInt128(0)   # stream 1 window over bits [64·(bitpos>>6), +128)
    w2 = UInt128(0)   # stream 2 window; bitpos = i·b throughout
    # scalar head through coefficient 7: covers the i = 0 wraparound so the
    # vector loop's descending 8-loads stay contiguous
    i = min(8, nconv)
    fp_unpack_scalar!(wordbuf, x1, x2, N, 0, i, 0, b, n1, n1p, n2, n2p)
    w1, w2, nw = fp_accum_words!(r, ro, s2, nw, w1, w2, 0, b, wordbuf, i)
    @inbounds while i + 8 <= nconv
        v1, u = fp_garner(SIMD.shufflevector(SIMD.vload(VF8, x1, N - i - 6), REV8),
                          SIMD.shufflevector(SIMD.vload(VF8, x2, N - i - 6), REV8),
                          n1, n1p, n2, n2p)
        c1v = reinterpret(V8, v1 + FP_MAGIC) & mask
        uv = reinterpret(V8, u + FP_MAGIC) & mask
        sv = (UInt64(i) * ub + vlane) & UInt64(63)
        svc = UInt64(63) - sv
        SIMD.vstore(c1v << sv, wordbuf, 1)
        SIMD.vstore((c1v >> 1) >> svc, wordbuf, 9)
        SIMD.vstore(uv << sv, wordbuf, 17)
        SIMD.vstore((uv >> 1) >> svc, wordbuf, 25)
        w1, w2, nw = fp_accum_words!(r, ro, s2, nw, w1, w2, i * b, b, wordbuf, 8)
        if nw >= 8
            w1, fold = fp_fold_s2!(r, ro, s2, nw, fold, w1)
            nw = 0
        end
        i += 8
    end
    # scalar tail: the final nconv mod 8 coefficients
    tail = nconv - i
    fp_unpack_scalar!(wordbuf, x1, x2, N, i, tail, i * b, b, n1, n1p, n2, n2p)
    w1, w2, nw = fp_accum_words!(r, ro, s2, nw, w1, w2, i * b, b, wordbuf, tail)
    @inbounds for outw in (nconv*b)>>6+1:rn
        r[ro+outw] = w1 % UInt64
        nw += 1
        s2[nw] = w2 % UInt64
        w1 >>= 64
        w2 >>= 64
    end
    fp_fold_s2!(r, ro, s2, nw, fold, w1)
    return r
end

# Pick the smallest admissible N = m·2^k >= T.  The multiplier set is tiered
# by T: while the FFT fits in L1/L2 only very fast transforms are worthwhile;
# in L3 we are memory bound, but running several slow odd-radix stages to save
# <1% of the points still isn't worth it — the full set waits for RAM sizes.
# A pure 2^k tops out at 2^34 (~37 GB operands) since 2^k must divide all p-1.
# The odd multipliers then extend reach to 315·2^34 (~12 TB).
const NTT_MS_SMALL = [1, 3, 5, 9]
const NTT_MS_MID   = [1, 3, 5, 7, 9, 15, 21, 105]
const NTT_MS_LARGE = [1, 3, 5, 7, 9, 15, 21, 35, 45, 63, 105, 315]
function ntt_len(T::Int, ctxs::Vararg{FpCtx,N}) where {N}
    maxk = minimum(two_adicity(F) for F in ctxs)
    best = typemax(Int)
    ms = T <= 1 << 14 ? NTT_MS_SMALL : # L1/L2
         T <= 1 << 20 ? NTT_MS_MID :   # L3
                        NTT_MS_LARGE   # RAM
    for m in ms
        k = max(Base.top_set_bit(cld(T, m) - 1), 2)   # ceil(log2), >= 4 points
        k <= maxk || continue                         # 2^k must divide p-1
        c = m << k
        c < best && (best = c)
    end
    best == typemax(Int) &&
        throw(ArgumentError("fp NTT length $T exceeds the 315·2^$maxk the primes support"))
    return best
end

# Chunk width and transform length against the CRT modulus prod(primes).
# The bound min(nca,ncb)·(2^b-1)^2 < prod(primes) is checked in division form:
# the product overflows UInt128 at large b with large operands.  b must stay
# below every prime so one pack serves both residues, and the search descends
# from an upper estimate of the answer — (2^b-1)² <= pp/min_nc caps b at
# (L_pp - L_nc + 3)/2, using the fewest-chunks (b = maxb) lower bound for
# min_nc — so it settles within a couple of UInt128 divides.  (This runs per
# multiplication and was ~40% of a 224-limb mul when the loop rebuilt the
# UInt128 product, allocating, every iteration.)
# Vararg{FpCtx,N} (not FpCtx...) forces specialization on the concrete ctx
# types: the unspecialized form dynamic-dispatches every fp_prime call.
function fp_ntt_params(bits_a::Int, bits_b::Int, ctxs::Vararg{FpCtx,N}) where {N}
    pp = prod(UInt128 ∘ fp_prime, ctxs) - 1
    maxb = Base.top_set_bit(minimum(fp_prime.(ctxs))) - 1
    nclow = Base.top_set_bit(cld(min(bits_a, bits_b), maxb))
    bstart = min(maxb, (Base.top_set_bit(pp) - nclow + 3) >> 1)
    for b in bstart:-1:1
        nca = cld(bits_a, b)
        ncb = cld(bits_b, b)
        if min(nca, ncb) <= pp ÷ UInt128(2^b - 1)^2
            return b, ntt_len(nca + ncb - 1, ctxs...)
        end
    end
    error("unreachable: b == 1 always satisfies the bound for supported sizes")
end

# Two-prime CRT pipeline, the mpn-layer entry points: r[1..m+n] = a[1..m]·b[1..n],
# r must not alias the inputs.  The working modulus p1·p2 ≈ 2^99.99 gives chunk
# width b ≈ (100 - log2 nc)/2: 46 bits at 256 chunks, >32 until >10 billion chunks.
function mul_fpntt2!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, m::Int,
                     b::Memory{Limb}, bo::Int, n::Int)
    bits_a = magnitude_bits(a, ao, m)
    bits_b = magnitude_bits(b, bo, n)
    bch, N = fp_ntt_params(bits_a, bits_b, FP_CTX1, FP_CTX2)
    plan1 = fp_ntt_plan(N, FP_CTX1)
    plan2 = fp_ntt_plan(N, FP_CTX2)
    nca, ncb = cld(bits_a, bch), cld(bits_b, bch)
    # a is packed once (tmp3) and kept for the in-place plan-2 transform; only
    # b, whose plan-1 copy is consumed by the pointwise product, is re-packed.
    # One backing allocation wrapped as three dense Vectors: one malloc instead
    # of three (small sizes sit right at the Karatsuba crossover), without the
    # per-access indexing overhead views cost in the radix loops.
    buf = Memory{Float64}(undef, 3N)
    tmp1 = Base.wrap(Array, memoryref(buf, 1), (N,))
    tmp2 = Base.wrap(Array, memoryref(buf, N + 1), (N,))
    tmp3 = Base.wrap(Array, memoryref(buf, 2N + 1), (N,))
    fp_ntt_pack!(tmp3, a, ao, m, bch, nca)
    copyto!(tmp1, tmp3)
    fp_ntt_fwd!(tmp1, plan1)
    fp_ntt_pack!(tmp2, b, bo, n, bch, ncb)
    fp_ntt_fwd!(tmp2, plan1)
    fp_ntt_pointwise!(tmp1, tmp2, FP_CTX1)
    fp_ntt_rev!(tmp1, plan1)
    fp_ntt_fwd!(tmp3, plan2)
    fp_ntt_pack!(tmp2, b, bo, n, bch, ncb)
    fp_ntt_fwd!(tmp2, plan2)
    fp_ntt_pointwise!(tmp3, tmp2, FP_CTX2)
    fp_ntt_rev!(tmp3, plan2)
    fp_ntt_unpack2!(r, ro, m + n, tmp1, tmp3, nca + ncb - 1, bch,
                    plan1.ninv, plan1.ninvp, plan2.ninv, plan2.ninvp)
    return nothing
end

function sqr_fpntt2!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, n::Int)
    bits = magnitude_bits(a, ao, n)
    bch, N = fp_ntt_params(bits, bits, FP_CTX1, FP_CTX2)
    plan1 = fp_ntt_plan(N, FP_CTX1)
    plan2 = fp_ntt_plan(N, FP_CTX2)
    nca = cld(bits, bch)
    # single pack: tmp2 keeps the packed input for the in-place plan-2 pass
    buf = Memory{Float64}(undef, 2N)
    tmp1 = Base.wrap(Array, memoryref(buf, 1), (N,))
    tmp2 = Base.wrap(Array, memoryref(buf, N + 1), (N,))
    fp_ntt_pack!(tmp2, a, ao, n, bch, nca)
    copyto!(tmp1, tmp2)
    fp_ntt_fwd!(tmp1, plan1)
    fp_ntt_pointwise!(tmp1, tmp1, FP_CTX1)
    fp_ntt_rev!(tmp1, plan1)
    fp_ntt_fwd!(tmp2, plan2)
    fp_ntt_pointwise!(tmp2, tmp2, FP_CTX2)
    fp_ntt_rev!(tmp2, plan2)
    fp_ntt_unpack2!(r, ro, 2n, tmp1, tmp2, 2nca - 1, bch,
                    plan1.ninv, plan1.ninvp, plan2.ninv, plan2.ninvp)
    return nothing
end
