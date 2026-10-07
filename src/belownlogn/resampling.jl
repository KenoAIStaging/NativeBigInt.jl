# Gaussian resampling with the permutations left in the transform, the
# chirp, and the complex convolution by a coefficient twist (paper,
# Sections 8 and 9.4–9.5).
#
# For coprime s < t (s an odd prime, t a power of two) with θ = t/s - 1,
# the Harvey–van der Hoeven resampling identity reads, in the paper's form
# with both coordinate permutations retained,
#
#     P_s F_s = 2^(2α²) B_0 P_t F_t A,     (P_s u)_j = u_{t j},  (P_t v)_k = v_{-s k},
#
# with contractions A = S/2 (Gaussian expansion to length t) and
# B_0 = D' J' C (selection of the nearest coordinates, the Neumann inverse of
# N = C T D, and a diagonal scale).  Tensored over the d axes this gives
# R F_s = 2^γ B_0 Q F_t A with γ = 2 d α², and Q F_t is Bluestein's chirp:
# Q F_t u = ā · (a * (ā · u)) / T.  The cyclic convolution on the power-of-two
# box is the synthetic ring convolution of synthetic.jl after twisting the
# last coordinate by ζ^k, ζ = e^(π i / r).
#
# All tables (Gaussian weights, chirp phases, roots) are computed once per
# axis with MPFR at p + 32 bits and truncated toward zero to the 2^-p grid,
# so every table value is in the disk with error below √2 2^-p; the paper
# charges O(p^(1+δ)) per such value through the established multiplier and
# needs nothing more precise than that.

# --- grid constants ------------------------------------------------------------------

const TABLE_GUARD = 32

# truncate x·2^F toward zero
grid_trunc(::Type{FW{L}}, x::BigFloat, F::Int) where {L} = FW{L}(trunc(BigInt, x * big(2)^F))

# e^(2πi num/den) on the p-bit disk grid
function grid_root(::Type{GC{L}}, num::Integer, den::Integer, p::Int) where {L}
    setprecision(BigFloat, p + TABLE_GUARD) do
        θ = 2 * BigFloat(pi) * BigFloat(mod(num, den)) / BigFloat(den)
        GC{L}(grid_trunc(FW{L}, cos(θ), p), grid_trunc(FW{L}, sin(θ), p))
    end
end

# --- one axis -------------------------------------------------------------------------

struct AxisMaps{L}
    s::Int
    t::Int
    p::Int
    F::Int                      # fractional bits of the weight tables (p + guard)
    α::Int
    RS::Int                     # expansion window radius in source indices
    mE::Int                     # Neumann window radius
    K::Int                      # Neumann terms
    T1::Vector{FW{L}}           # exp(-π u²/(α t)²)/(2α) for u_c = -(t/2)+1 .. t/2  (index u_c + t/2)
    T2::Vector{FW{L}}           # exp(-π (1 + 2u/t)/α²) for u = -t/2 .. t/2        (index u + t/2 + 1)
    H::FW{L}                    # exp(-2π/α²)
    TE::Vector{FW{L}}           # off-diagonal weights of N - I as a function of x = v_j + t δ:
                                # exp(-π α² (x² - red(x)²)/s²), red(x) = x - s⌊x/s + 1/2⌋, |x| ≤ (s-1)/2 + t mE
    xoff::Int                   # index offset: TE[x + xoff]
    Dp::Vector{FW{L}}           # 2^-(2α²-2) exp(π α² v_j²/s²) at p bits, j = 0..s-1
    q::Vector{Int}              # q_j = ⌊t j/s + 1/2⌋
    v::Vector{Int}              # v_j = t j - s q_j ∈ (-s/2, s/2]
end

function AxisMaps{L}(s::Int, t::Int, p::Int, α::Int) where {L}
    isodd(s) && s >= 3 && ispow2(t) && s < t || throw(ArgumentError("need an odd prime s < t = 2^k"))
    F = min(p + TABLE_GUARD, 64L - 4)
    θ = t / s - 1
    RS = ceil(Int, sqrt(p)) * α
    mE = ceil(Int, sqrt(p) / (2α)) + 1
    K = ceil(Int, p / (α^2 * θ))
    q = [fld(t * j, s) + ((2 * mod(t * j, s) >= s) ? 1 : 0) for j in 0:s-1]
    v = [t * j - s * q[j+1] for j in 0:s-1]
    hs = (s - 1) ÷ 2
    T1 = Vector{FW{L}}(undef, t)
    T2 = Vector{FW{L}}(undef, t + 1)
    xmax = hs + t * mE
    TE = Vector{FW{L}}(undef, 2xmax + 1)
    xoff = xmax + 1
    Dp = Vector{FW{L}}(undef, s)
    H = setprecision(BigFloat, p + TABLE_GUARD) do
        π_ = BigFloat(pi)
        a2 = BigFloat(α)^2
        for uc in (-(t ÷ 2) + 1):(t ÷ 2)
            T1[uc+t÷2] = grid_trunc(FW{L}, exp(-π_ * BigFloat(uc)^2 / (a2 * BigFloat(t)^2)) / (2α), F)
        end
        for u in -(t ÷ 2):(t ÷ 2)
            T2[u+t÷2+1] = grid_trunc(FW{L}, exp(-π_ * (1 + 2 * BigFloat(u) / t) / a2), F)
        end
        for x in -xmax:xmax
            red = x - s * fld(2x + s, 2s)          # x - s⌊x/s + 1/2⌋ ∈ (-s/2, s/2]
            e = π_ * a2 * (BigFloat(x)^2 - BigFloat(red)^2) / BigFloat(s)^2
            TE[x+xoff] = grid_trunc(FW{L}, exp(-e), F)
        end
        for j in 0:s-1
            x = π_ * a2 * BigFloat(v[j+1])^2 / BigFloat(s)^2
            Dp[j+1] = grid_trunc(FW{L}, exp(x) / big(2)^(2α^2 - 2), p)
        end
        grid_trunc(FW{L}, exp(-2π_ / a2), F)
    end
    AxisMaps{L}(s, t, p, F, α, RS, mE, K, T1, T2, H, TE, xoff, Dp, q, v)
end

# truncated product of two F-bit fixed-point fractions
@inline function mulF(a::FW{L}, b::FW{L}, F::Int, ::Type{FW{L2}}) where {L,L2}
    resize(FW{L}, trunc_shr(resize(FW{L2}, a) * resize(FW{L2}, b), F))
end

# Expansion Ã = S̃': the s values at `src` (stride `st`) to t values at `dst`.
function expand_line!(dst::AbstractVector{GC{L}}, src::AbstractVector{GC{L}}, ax::AxisMaps{L}) where {L}
    s, t, F = ax.s, ax.t, ax.F
    L2 = 2L + 1
    W2 = FW{L2}
    @inbounds for k in 0:t-1
        # center j_c with u_c = t j_c - s k ∈ (-t/2, t/2]
        jc = fld(s * k, t)
        uc = t * jc - s * k
        if uc <= -(t ÷ 2)
            jc += 1; uc += t
        end
        accr = zero(W2); acci = zero(W2)
        # rightward from the center
        w = ax.T1[uc+t÷2]
        g = ax.T2[uc+t÷2+1]
        j = jc
        while j <= jc + ax.RS && !iszero(w)
            z = src[mod(j, s)+1]
            accr += resize(W2, w) * resize(W2, z.re)
            acci += resize(W2, w) * resize(W2, z.im)
            w = mulF(w, g, F, W2)
            g = mulF(g, ax.H, F, W2)
            j += 1
        end
        # leftward: ratios exp(-(A(2i+1) - B)) start at T2[-u_c]
        g = ax.T2[-uc+t÷2+1]
        w = mulF(ax.T1[uc+t÷2], g, F, W2)
        g = mulF(g, ax.H, F, W2)
        j = jc - 1
        while j >= jc - ax.RS && !iszero(w)
            z = src[mod(j, s)+1]
            accr += resize(W2, w) * resize(W2, z.re)
            acci += resize(W2, w) * resize(W2, z.im)
            w = mulF(w, g, F, W2)
            g = mulF(g, ax.H, F, W2)
            j -= 1
        end
        dst[k+1] = GC{L}(resize(FW{L}, trunc_shr(accr, F)), resize(FW{L}, trunc_shr(acci, F)))
    end
    return dst
end

# Ẽ z for z ∈ D_p^s, truncated to the grid.  N = C T D has entries
# exp(-π α² (t j'/s - q_j)²) e^{π α² β_{j'}²} at (j, j'); the diagonal is 1 and
# the entry at j' = j + δ is W(v_j + t δ) with W(x) = exp(-π α² (x² - red(x)²)/s²).
function neumann_step!(out::Vector{GC{L}}, z::Vector{GC{L}}, ax::AxisMaps{L}) where {L}
    s, F = ax.s, ax.F
    L2 = 2L + 1
    W2 = FW{L2}
    @inbounds for j in 0:s-1
        vj = ax.v[j+1]
        accr = zero(W2); acci = zero(W2)
        for δ in 1:ax.mE
            wplus = ax.TE[vj+ax.t*δ+ax.xoff]
            wminus = ax.TE[vj-ax.t*δ+ax.xoff]
            zp = z[mod(j + δ, s)+1]
            zm = z[mod(j - δ, s)+1]
            accr += resize(W2, wplus) * resize(W2, zp.re) + resize(W2, wminus) * resize(W2, zm.re)
            acci += resize(W2, wplus) * resize(W2, zp.im) + resize(W2, wminus) * resize(W2, zm.im)
        end
        out[j+1] = GC{L}(resize(FW{L}, trunc_shr(accr, F)), resize(FW{L}, trunc_shr(acci, F)))
    end
    return out
end

# Compression B̃_0 = D̃' J̃' C: the t values at `src` to s values at `dst`.
function compress_line!(dst::AbstractVector{GC{L}}, src::AbstractVector{GC{L}}, ax::AxisMaps{L},
                        bufs::NTuple{3,Vector{GC{L}}}) where {L}
    s, p = ax.s, ax.p
    cur, nxt, acc = bufs
    # C then the first Neumann term u/2
    @inbounds for j in 0:s-1
        cur[j+1] = trunc_shr(src[ax.q[j+1]+1], 1)
        acc[j+1] = cur[j+1]
    end
    sign = -1
    for _ in 2:ax.K
        neumann_step!(nxt, cur, ax)
        @inbounds for j in 1:s
            acc[j] = sign == 1 ? acc[j] + nxt[j] : acc[j] - nxt[j]
        end
        cur, nxt = nxt, cur
        sign = -sign
    end
    L2 = GC{2L + 1}
    @inbounds for j in 0:s-1
        dst[j+1] = mul_trunc(acc[j+1], ax.Dp[j+1], p, L2)
    end
    return dst
end

# --- the tensor interface on the padded box ----------------------------------------------

# A padded box ∏[0, t_i) stored lexicographically, axis 1 most significant.
struct Box
    t::Vector{Int}
    s::Vector{Int}
    strides::Vector{Int}
    size::Int
end
function Box(t::Vector{Int}, s::Vector{Int})
    d = length(t)
    strides = [prod(t[i+1:end]; init=1) for i in 1:d]
    Box(t, s, strides, prod(t))
end

# Apply the one-dimensional map along axis i to every line whose other
# coordinates are below the given limits (validity), reading `nin` values
# and writing `nout`; lines invalid elsewhere are already zero.
function along_axis!(box::Box, A::Vector{GC{L}}, i::Int, limits::Vector{Int}, nin::Int, nout::Int, f!) where {L}
    d = length(box.t)
    st = box.strides[i]
    linein = Vector{GC{L}}(undef, nin)
    lineout = Vector{GC{L}}(undef, nout)
    # enumerate the other coordinates
    others = [k for k in 1:d if k != i]
    nlines = prod(box.t[k] for k in others; init=1)
    for ℓ in 0:nlines-1
        base = 0
        valid = true
        rem = ℓ
        for k in reverse(others)
            c = rem % box.t[k]
            rem ÷= box.t[k]
            c < limits[k] || (valid = false)
            base += c * box.strides[k]
        end
        valid || continue
        @inbounds for j in 0:nin-1
            linein[j+1] = A[base+j*st+1]
        end
        f!(lineout, linein)
        @inbounds for j in 0:nout-1
            A[base+j*st+1] = lineout[j+1]
        end
        @inbounds for j in nout:box.t[i]-1
            A[base+j*st+1] = zero(GC{L})
        end
    end
    return A
end

# Ã on the box: expand the axes one at a time, source coordinates valid below
# s_i until expanded, then below t_i.
function expand_box!(A::Vector{GC{L}}, box::Box, axes::Vector{AxisMaps{L}}) where {L}
    limits = copy(box.s)
    for i in eachindex(box.t)
        along_axis!(box, A, i, limits, box.s[i], box.t[i], (o, l) -> expand_line!(o, l, axes[i]))
        limits[i] = box.t[i]
    end
    return A
end

# B̃_0 on the box: compress the axes one at a time.
function compress_box!(A::Vector{GC{L}}, box::Box, axes::Vector{AxisMaps{L}}) where {L}
    limits = copy(box.t)
    for i in eachindex(box.t)
        s = box.s[i]
        bufs = (Vector{GC{L}}(undef, s), Vector{GC{L}}(undef, s), Vector{GC{L}}(undef, s))
        along_axis!(box, A, i, limits, box.t[i], s, (o, l) -> compress_line!(o, l, axes[i], bufs))
        limits[i] = s
    end
    return A
end

# --- chirp, twist and complex convolution ----------------------------------------------------

struct ChirpData{L}
    box::Box
    p::Int
    r::Int
    zeta::Vector{GC{L}}          # ζ^k, k = 0..2r-1, ζ = e^(πi/r)
    chirp_exp::Vector{Int}       # exponent of ζ in a_j = ζ^(-Σ s_i j_i² r/t_i mod 2r)
    kernel_t::Vector{GC{L}}      # synthetic forward transform of ι(θ⁺·ã)
    pl::SynthPlan
end

function ChirpData{L}(box::Box, pl::SynthPlan, p::Int) where {L}
    d = length(box.t)
    r = box.t[d]
    r == pl.r || throw(ArgumentError("last axis must have length r"))
    zeta = [grid_root(GC{L}, k, 2r, p) for k in 0:2r-1]
    ce = Vector{Int}(undef, box.size)
    for idx in 0:box.size-1
        E = 0
        rem = idx
        for i in d:-1:1
            ji = rem % box.t[i]
            rem ÷= box.t[i]
            E -= box.s[i] * ji * ji * (r ÷ box.t[i])
        end
        ce[idx+1] = mod(E, 2r)
    end
    kern = [zeta[ce[idx+1]+1] for idx in 0:box.size-1]
    twist!(kern, r, zeta, +1, p)
    synth_forward!(kern, pl)
    ChirpData{L}(box, p, r, zeta, ce, kern, pl)
end

# multiply the coefficient at last-axis index k by ζ^(±k), truncating
function twist!(A::Vector{GC{L}}, r::Int, zeta::Vector{GC{L}}, dir::Int, p::Int) where {L}
    L2 = GC{2L + 1}
    @inbounds for idx in 0:length(A)-1
        k = idx % r
        k == 0 && continue
        θ = dir == 1 ? zeta[k+1] : zeta[2r-k+1]
        A[idx+1] = mul_trunc(A[idx+1], θ, p, L2)
    end
    return A
end

# x ↦ Q_p(ā · x) with the chirp phases
function chirp_phase!(A::Vector{GC{L}}, cd::ChirpData{L}) where {L}
    L2 = GC{2L + 1}
    @inbounds for idx in eachindex(A)
        A[idx] = mul_trunc(A[idx], conj(cd.zeta[cd.chirp_exp[idx]+1]), cd.p, L2)
    end
    return A
end

# The computed chirp transform C̃(x) = Q_p(ā · M̃_C(ã, Q_p(ā · x))), whose exact
# counterpart is Q F_t; M̃_C is the twisted synthetic convolution with the
# kernel's transform cached.
function chirp_transform!(x::Vector{GC{L}}, cd::ChirpData{L}) where {L}
    chirp_phase!(x, cd)
    twist!(x, cd.r, cd.zeta, +1, cd.p)
    synth_convolution!(x, cd.kernel_t, cd.pl; g_transformed=true)
    twist!(x, cd.r, cd.zeta, -1, cd.p)
    chirp_phase!(x, cd)
    return x
end

# The source transform F̃ = 2^γ B̃_0 C̃ Ã, exact counterpart R F_s.
function source_transform!(x::Vector{GC{L}}, box::Box, axes::Vector{AxisMaps{L}}, cd::ChirpData{L}, γ::Int) where {L}
    expand_box!(x, box, axes)
    chirp_transform!(x, cd)
    compress_box!(x, box, axes)
    @inbounds for i in eachindex(x)
        x[i] = x[i] << γ
    end
    return x
end
# The opposite-sign transform F̃⁺(z) = conj(F̃(conj z)), exact counterpart R F_s^+.
function source_transform_plus!(x::Vector{GC{L}}, box::Box, axes::Vector{AxisMaps{L}}, cd::ChirpData{L}, γ::Int) where {L}
    @inbounds for i in eachindex(x); x[i] = conj(x[i]); end
    source_transform!(x, box, axes, cd, γ)
    @inbounds for i in eachindex(x); x[i] = conj(x[i]); end
    return x
end
