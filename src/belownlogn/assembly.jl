# Exact integer multiplication and its parameters (paper, Section 9).
#
# The algorithm, for two n-bit integers:
#   1. radix-2^b digits, scaled by 2^-(b+2), placed on the padded box through
#      the Chinese-remainder axis map Φ (Agarwal–Cooley);
#   2. two source transforms F̃ (resampling.jl);
#   3. pointwise product, truncated to the grid;
#   4. the opposite transform F̃⁺ and the exact scale by S;
#   5. the exact scale by 2^(2b+4) S, rounding to integers, Φ^-1;
#   6. radix-2^b carry propagation.
# The two scales by S undo the two normalizations left by the three source
# transforms ((u * v)/S² → u * v), and 2^(2b+4) undoes the two input scales.
#
# Parameters.  The paper fixes b = ⌈log2 n⌉, p = 6b, d = ⌊b^ε⌋ with
# ε = 1/40 (Colkitt's parameters; the preprint had ε = 2^-75), T ∈ [4n/b,
# 8n/b), r = 2^⌈log2(T)/d⌉, prime source lengths
# s_i ∈ ((1-2η) t_i, (1-η) t_i] with η = 1/(4d), α = ⌈(12 d² b)^(1/4)⌉ and
# γ = 2 d α², and proves every bound for all sufficiently large n.  Here
# every input length has to work, so:
#   * d follows the paper's formula (which is 1 for every b < 2^40, i.e. for
#     every n below 2^(2^40) bits) unless overridden, and is clamped so that
#     every axis has length ≥ 2;
#   * the prime intervals are widened downward when the paper's interval is
#     empty at small t, and T is doubled if the product S then fails to cover
#     the product degree;
#   * p is the smallest precision for which the paper's explicit error chain
#     (final error < 1/2, plus its disk margins) holds, which coincides with
#     6b once γ + O(log b) < b, i.e. for astronomically large n, and is a
#     few times b before that.

struct Params
    n::Int
    b::Int           # digit bits
    q::Int           # digits per operand
    p::Int           # grid precision
    d::Int           # axes
    T::Int           # padded box size
    r::Int           # last axis length (power of two)
    t::Vector{Int}   # axis lengths, each r/2 or r, t[d] = r
    s::Vector{Int}   # prime source lengths
    S::Int           # ∏ s_i
    α::Int
    γ::Int
    L::Int           # limbs per grid numerator
    paper_choice::Bool   # true when p and the primes follow the paper's formulas verbatim
end

lg(n::Int) = max(ndigits(n - 1, base=2), 1)   # ⌈log2 n⌉ for n ≥ 2; 1 for n = 1

function isprime_small(n::Int)
    n < 2 && return false
    n < 4 && return true
    iseven(n) && return false
    i = 3
    while i * i <= n
        n % i == 0 && return false
        i += 2
    end
    return true
end

# d = ⌊b^(1/40)⌋, found as the paper prescribes: binary search with exact
# comparisons d^40 ≤ b.  Works for any Integer b ≥ 1 and returns its type.
function paper_dimension(b::Integer)
    b >= 1 || throw(ArgumentError("b must be positive"))
    lo = one(b)
    hi = one(b) << cld(Base.top_set_bit(b), 40)     # lo^40 ≤ b < hi^40
    while hi - lo > 1
        mid = (lo + hi) >> 1
        if big(mid)^40 <= b
            lo = mid
        else
            hi = mid
        end
    end
    return lo
end

# Exact test d^num ≥ k^den for integers d ≥ 1, k ≥ 1, num, den ≥ 1: the
# floor of log_k d decides everything outside a thin band, and the band is
# settled by exact powers.  This is how fixed rational powers such as
# K = ⌊d^c⌋ ≥ 6 with c = 9/10^12 are compared without ever forming d^c.
function pow_ge(d::Integer, num::Integer, k::Integer, den::Integer)
    d >= 1 && k >= 1 && num >= 1 && den >= 1 || throw(ArgumentError("positive arguments required"))
    k == 1 && return true
    e = ndigits(d, base=k) - 1                       # ⌊log_k d⌋, so k^e ≤ d < k^(e+1)
    num * e >= den && return true                    # d^num ≥ k^(num e) ≥ k^den
    num * (e + 1) <= den && return false             # d^num < k^(num (e+1)) ≤ k^den
    return big(d)^num >= big(k)^den
end

# Distinct odd primes near the axis lengths; returns nothing if some axis has
# no admissible prime at all.
function choose_primes(t::Vector{Int}, d::Int)
    used = Set{Int}()
    s = Int[]
    paper = true
    for ti in t
        hi = fld((4d - 1) * ti, 4d)               # ⌊(1-η) t⌋
        lo_paper = fld((4d - 2) * ti, 4d)         # (1-2η) t, exclusive
        found = 0
        for c in hi:-1:lo_paper+1
            if isodd(c) && c ∉ used && isprime_small(c)
                found = c; break
            end
        end
        if found == 0
            paper = false
            for c in hi:-1:(ti ÷ 2 + 1)
                if isodd(c) && c >= 3 && c ∉ used && isprime_small(c)
                    found = c; break
                end
            end
        end
        found == 0 && return nothing, false
        push!(s, found); push!(used, found)
    end
    return s, paper
end

# The paper's explicit error chain in units of 2^-p: E_s = 2^γ (2 d p² + 8 T L),
# E_w = S (3 E_s + 2), final error 2^(2b+4) S E_w 2^-p.  Also its disk margins.
function error_chain_ok(p::Int, b::Int, d::Int, T::Int, S::Int, γ::Int)
    Lg = ndigits(T - 1, base=2)
    hp = 1 // big(2)^p
    Es = big(2)^γ * (2 * d * big(p)^2 + 8 * big(T) * Lg)
    Ew = big(S) * (3Es + 2)
    final = big(2)^(2b + 4) * big(S) * Ew * hp
    final < 1 // 2 || return false
    margin = 1 // 1000
    d * big(p)^2 * hp < margin || return false
    8 * big(T) * Lg * hp < margin || return false
    Es * hp < margin || return false
    28 * big(2)^γ * big(T)^2 * Lg * hp < margin || return false
    return true
end

function Params(n::Int; d::Union{Nothing,Int}=nothing)
    n >= 1 || throw(ArgumentError("n must be positive"))
    b = lg(n)
    q = cld(n, b)
    # d = ⌊b^ε⌋ with ε = 1/40, which is 1 for every b < 2^40
    d_req = d === nothing ? Int(paper_dimension(b)) : d
    d_req >= 1 || throw(ArgumentError("d must be positive"))
    T = 1 << ndigits(cld(4n, b) - 1, base=2)      # power of two in [4n/b, 8n/b)
    while true
        lT = trailing_zeros(T)
        dd = min(d_req, max(1, lT ÷ 2))            # every axis length ≥ 4 when d > 1
        ell = cld(lT, dd)
        r = 1 << ell
        nshort = dd * ell - lT                      # in [0, dd): these axes get r/2
        t = vcat(fill(r ÷ 2, nshort), fill(r, dd - nshort))
        if dd > 1 && r ÷ 2 < 4
            T <<= 1; continue
        end
        s, primes_paper = choose_primes(t, dd)
        if s === nothing
            T <<= 1; continue
        end
        S = prod(s)
        if !(S > 2q - 2)                            # no wraparound of the degree-(2q-2) product
            T <<= 1; continue
        end
        α = ceil(Int, (12 * dd^2 * b)^(1 / 4))
        while α^4 < 12 * dd^2 * b; α += 1; end
        γ = 2 * dd * α^2
        p = 6b
        while !error_chain_ok(p, b, dd, T, S, γ)
            p += max(1, p ÷ 16)
        end
        # width: grid numerators ≤ 2^p plus the exact scales by S and 2^(2b+4) S
        L = cld(p + 4b + 16, 64)
        return Params(n, b, q, p, dd, T, r, t, s, S, α, γ, L, primes_paper && p == 6b)
    end
end

# --- the Chinese-remainder axis map ---------------------------------------------------------
#
# Φ: C[x]/(x^S - 1) → ⊗ C[x_i]/(x_i^{s_i} - 1), x ↦ ∏ x_i^{μ_i} with
# μ_i = P_i^-1 mod s_i, P_i = ∏_{j<i} s_j.  Coefficient k lands at the
# tensor coordinates b_i = μ_i k mod s_i; the paper performs this as a
# sequence of controlled cyclic shifts (b_i = a_i + μ_i Σ_{j<i} a_j P_j) on
# the mixed-radix digits a_j of k, which on a RAM is just the formula.

struct CRTMap
    s::Vector{Int}
    P::Vector{Int}      # prefix products
    μ::Vector{Int}
    strides::Vector{Int}
end
function CRTMap(box::Box)
    d = length(box.s)
    P = [prod(box.s[1:i-1]; init=1) for i in 1:d]
    μ = [invmod(P[i] % box.s[i], box.s[i]) for i in 1:d]
    CRTMap(box.s, P, μ, box.strides)
end
# flat box address of coefficient k
@inline function crt_address(m::CRTMap, k::Int)
    idx = 0
    @inbounds for i in eachindex(m.s)
        idx += mod(m.μ[i] * k, m.s[i]) * m.strides[i]
    end
    return idx
end
# coefficient index of the valid box coordinates b (mixed-radix recovery)
function crt_index(m::CRTMap, bcoords::Vector{Int})
    k = 0
    @inbounds for i in eachindex(m.s)
        ai = mod(bcoords[i] - m.μ[i] * (k % m.s[i]), m.s[i])
        k += ai * m.P[i]
    end
    return k
end

# --- the multiplication ------------------------------------------------------------------------

# bits [lo, lo+cnt) of the limb array as an Int (cnt ≤ 63)
@inline function getbits(a::Memory{Limb}, ao::Int, nlimbs::Int, lo::Int, cnt::Int)
    li, bi = divrem(lo, 64)
    li += 1
    w = li <= nlimbs ? (@inbounds a[ao+li]) >> bi : zero(Limb)
    if bi + cnt > 64 && li + 1 <= nlimbs
        w |= (@inbounds a[ao+li+1]) << (64 - bi)
    end
    return Int(w & ((one(Limb) << cnt) - 1))
end

# radix-2^b digits of an n-bit operand as grid values a_j 2^-(b+2), placed on
# the box through Φ; zero on the padding
function place_digits!(A::Vector{GC{L}}, a::Memory{Limb}, ao::Int, na::Int, prm::Params, crt::CRTMap) where {L}
    fill!(A, zero(GC{L}))
    shift = prm.p - prm.b - 2
    @inbounds for j in 0:prm.q-1
        dig = getbits(a, ao, na, j * prm.b, prm.b)
        dig == 0 && continue
        A[crt_address(crt, j)+1] = GC{L}(FW{L}(dig) << shift, zero(FW{L}))
    end
    return A
end

# Append a nonnegative integer `coef` times 2^(b*k) worth of carries: the
# radix-2^b carry propagation writing bits into `out` (2n bits, zero-filled).
function carry_out!(out::Memory{Limb}, coeffs::Vector{Int128}, b::Int, n::Int)
    carry = zero(Int128)
    Q0 = Int128(1) << b
    nbits = 2n
    pos = 0
    for c in coeffs
        pos >= nbits && break
        v = c + carry
        dig = v & (Q0 - 1)
        carry = v >> b
        putbits!(out, pos, Int(dig), b)
        pos += b
    end
    while carry != 0 && pos < nbits
        dig = carry & (Q0 - 1)
        carry >>= b
        putbits!(out, pos, Int(dig), b)
        pos += b
    end
    return out
end
@inline function putbits!(out::Memory{Limb}, pos::Int, v::Int, cnt::Int)
    li, bi = divrem(pos, 64)
    li += 1
    nl = length(out)
    w = Limb(v)
    li <= nl && (@inbounds out[li] |= w << bi)
    if bi + cnt > 64 && li + 1 <= nl
        @inbounds out[li+1] |= w >> (64 - bi)
    end
    return out
end

struct MulPlan{L}
    prm::Params
    box::Box
    crt::CRTMap
    axes::Vector{AxisMaps{L}}
    pl::SynthPlan
    cd::ChirpData{L}
    ctx::LayerCtx
end

function MulPlan(prm::Params; ctx::Union{Nothing,LayerCtx}=nothing)
    L = prm.L
    return _mulplan(Val(L), prm, ctx)
end
function _mulplan(::Val{L}, prm::Params, ctx) where {L}
    ctx = ctx === nothing ? paper_layer_ctx(prm.d, prm.p) : ctx
    box = Box(prm.t, prm.s)
    crt = CRTMap(box)
    axes = [AxisMaps{L}(prm.s[i], prm.t[i], prm.p, prm.α) for i in 1:prm.d]
    pl = SynthPlan(prm.r, [trailing_zeros(x) for x in prm.t[1:end-1]], prm.p, ctx)
    cd = ChirpData{L}(box, pl, prm.p)
    MulPlan{L}(prm, box, crt, axes, pl, cd, ctx)
end

# r[ro+1 .. ro+2*nl] ← a[1..na] * b[1..nb] with both operands read as n-bit
# strings (n = prm.n), nl = cld(n, 64).  The result buffer must hold 2nl limbs.
function multiply!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, na::Int,
                   b::Memory{Limb}, bo::Int, nb::Int, mp::MulPlan{L}) where {L}
    prm = mp.prm
    n, bb, p, S = prm.n, prm.b, prm.p, prm.S
    γ = prm.γ
    u = Vector{GC{L}}(undef, mp.box.size)
    v = Vector{GC{L}}(undef, mp.box.size)
    place_digits!(u, a, ao, na, prm, mp.crt)                 # step 1
    place_digits!(v, b, bo, nb, prm, mp.crt)
    source_transform!(u, mp.box, mp.axes, mp.cd, γ)           # step 2
    source_transform!(v, mp.box, mp.axes, mp.cd, γ)
    L2 = GC{2L + 1}
    @inbounds for i in eachindex(u)                           # step 3
        u[i] = mul_trunc(u[i], v[i], p, L2)
    end
    source_transform_plus!(u, mp.box, mp.axes, mp.cd, γ)      # step 4
    # steps 4–5: exact scales by S, then 2^(2b+4) S, then round the real parts
    sh = 2bb + 4
    coeffs = zeros(Int128, S)
    bcoords = zeros(Int, prm.d)
    @inbounds for idx in 0:mp.box.size-1
        rem = idx
        valid = true
        for i in prm.d:-1:1
            c = rem % prm.t[i]
            rem ÷= prm.t[i]
            bcoords[i] = c
            c < prm.s[i] || (valid = false)
        end
        valid || continue
        x = u[idx+1].re * Int64(S)
        x = (x * Int64(S)) << sh
        c = Int128(round_shr(x, p))
        0 <= c < Int128(1) << (3bb) || error("recovered coefficient out of range: the error bound failed")
        coeffs[crt_index(mp.crt, bcoords)+1] = c
    end
    nl = cld(n, 64)
    fill!(view(r, ro+1:ro+2nl), zero(Limb))
    out = Memory{Limb}(undef, 2nl)
    fill!(out, zero(Limb))
    carry_out!(out, coeffs, bb, n)                            # step 6
    copyto!(r, ro + 1, out, 1, 2nl)
    return r
end

# Convenience: product of two nonnegative limb magnitudes as a normalized NBig.
function mul_belownlogn(x::NBig, y::NBig; d::Union{Nothing,Int}=nothing)
    (iszero(x) || iszero(y)) && return zero(NBig)
    la, lb = abs(x.signlen), abs(y.signlen)
    bits(z, l) = 64 * (l - 1) + Base.top_set_bit(z.limbs[l])
    n = max(bits(x, la), bits(y, lb))
    prm = Params(n; d)
    mp = MulPlan(prm)
    nl = cld(n, 64)
    r = Memory{Limb}(undef, 2nl)
    multiply!(r, 0, x.limbs, 0, la, y.limbs, 0, lb, mp)
    return nbig_from_limbs(sign(x.signlen) * sign(y.signlen), r, 2nl)
end

# mpn-style entry: r[ro+1 .. ro+m+n] = a[1..m] * b[1..n] for normalized
# nonzero magnitudes, m ≥ n ≥ 1.  Both operands are read as strings of
# max(bits) bits, as the paper's machine does.
function mul_limbs!(r::Memory{Limb}, ro::Int, a::Memory{Limb}, ao::Int, m::Int,
                    b::Memory{Limb}, bo::Int, n::Int; d::Union{Nothing,Int}=nothing)
    bits_a = 64 * (m - 1) + Base.top_set_bit(@inbounds a[ao+m])
    bits_b = 64 * (n - 1) + Base.top_set_bit(@inbounds b[bo+n])
    nbits = max(bits_a, bits_b)
    prm = Params(nbits; d)
    mp = MulPlan(prm)
    nl = cld(nbits, 64)
    tmp = Memory{Limb}(undef, 2nl)
    multiply!(tmp, 0, a, ao, m, b, bo, n, mp)
    copyto!(r, ro + 1, tmp, 1, m + n)
    return nothing
end
