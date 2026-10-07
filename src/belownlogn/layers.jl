# Fast simultaneous synthetic butterfly layers (paper, Section 5), in the
# RAM model.
#
# A layer applies H_0 = ½ [1 1; 1 -1] on D selected address bits of an array
# of Gaussian dyadics at once.  Writing C = H S H = aI + bX with a = (1+i)/2,
# b = (1-i)/2, the identity H_0^{⊗D} = b^D S_D C^{⊗D} S_D turns the layer into
# a C layer plus diagonal phases, and C^{⊗e} on a group of e = m f selected
# bits is computed by running the finite complex network of motifs.jl on
# W role streams whose frames are tensor powers of the phase frames
# C_U = H^{⊗m} diag(i^{wt(P_U x)}) H^{⊗m}.  Every edge between nested labels
# costs one child call C^{⊗f} on the f bits of one slot per vector of an
# orthonormal basis of its residual, so a node with e selected bits makes s
# child calls on streams of volume V/W, and s < Wm is the saving.
#
# What is tape-specific in the paper and what is not:
#   * the recurrence F(e) ≤ (s/W) F(e/m) + overhead, the gates, the frames and
#     the guard analysis are pure linear algebra and are implemented as
#     written;
#   * the packed selected-bit additions (Lemma 5.x) and the chunk swaps exist
#     only because address bits on a tape cannot be permuted for free.  Here
#     a change of binary basis P_M is a gather through a permutation table,
#     and the role split is a strided copy.
#
# The network constants decide when the recursion engages: the layer applies
# individual kernels when D ≤ q_0 = ⌈log_2 W⌉⌈log_m 2d⌉.  With the paper's
# h = 100 network (W ≈ 1.9·10^21, m = 10^6) that is q_0 = 71 · ⌈log_m 2d⌉,
# so every layer that fits in memory takes the individual-kernel path; the
# recursive path is exercised by the tests with small networks.

# Network constants visible to the layer: either an explicit LinearNetwork
# (small, runnable) or the closed-form counts of the paper's network.
struct LayerCtx
    net::Union{Nothing,LinearNetwork}
    W::BigInt
    m::BigInt
    s::BigInt
    gate_ops::BigInt          # elementary operations per invocation chain, excluding children
    d::Int                    # the family parameter d of the layer
    leaf::Int                 # stop the recursion below this many bits (paper: ⌈d^β⌉, β = 1/2)
    p::Int                    # grid precision of the arrays this context acts on
    onb_cache::Dict{Tuple{Int,Int},Tuple{Vector{F2Vec},Vector{Int}}}
end

function LayerCtx(net::LinearNetwork, d::Int, p::Int; leaf::Int=ceil(Int, sqrt(d)))
    s = residual_rank_sum(net)
    ops = 2 * gate_terms(net) + 4s + 4net.W + 4
    LayerCtx(net, big(net.W), big(net.m), big(s), big(ops), d, leaf, p,
             Dict{Tuple{Int,Int},Tuple{Vector{F2Vec},Vector{Int}}}())
end
function LayerCtx(counts::MotifCounts, d::Int, p::Int; leaf::Int=ceil(Int, sqrt(d)))
    counts.kind === :complex || throw(ArgumentError("the layer uses the complex network"))
    ops = 24counts.W^3 + 4counts.s + 4counts.W + 4
    LayerCtx(nothing, counts.W, big(counts.m), counts.s, ops, d, leaf, p,
             Dict{Tuple{Int,Int},Tuple{Vector{F2Vec},Vector{Int}}}())
end
# the paper's layer: the h = 100 complex network
const PAPER_NETWORK = motif_counts(100, :complex)
paper_layer_ctx(d::Int, p::Int) = LayerCtx(PAPER_NETWORK, d, p)

# number of selected bits processed individually before the rows are formed
function q0_bits(ctx::LayerCtx)
    lw = ndigits(ctx.W - 1, base=2)                      # ⌈log2 W⌉
    lm = ceil(Int, log(2 * ctx.d) / log(Float64(ctx.m)))  # ⌈log_m 2d⌉
    lm = max(lm, 1)
    # exact: smallest k with m^k ≥ 2d
    while ctx.m^lm < 2 * ctx.d; lm += 1; end
    while lm > 1 && ctx.m^(lm - 1) >= 2 * ctx.d; lm -= 1; end
    return lw * lm
end

# Dependency-depth bound of the C layer on e bits (paper, guard-width bound):
# A(e) ≤ s A(e/m) + E at an internal node, A(e) ≤ 8e at a leaf.
function depth_bound(ctx::LayerCtx, e::Int)
    if e < ctx.m || e < ctx.leaf || e % ctx.m != 0
        return big(8e)
    end
    return ctx.s * depth_bound(ctx, e ÷ Int(ctx.m)) + ctx.gate_ops
end

# Guard bits for a layer on D selected bits: depth of the C layer on the
# largest piece, the q_0 individual kernels, and the final b^D scale.
function guard_bits(ctx::LayerCtx, D::Int)
    q0 = q0_bits(ctx)
    rest = max(D - q0, 0)
    A = big(8 * min(D, q0))
    if rest > 0 && ctx.net !== nothing
        # base-m pieces of the remaining bits; their depths concatenate
        A += sum_piece_depth(ctx, rest)
    elseif rest > 0
        A += big(8rest)
    end
    A += D ÷ 2 + 9
    A > 1 << 20 && throw(ArgumentError("guard bound $A too large to instantiate"))
    return Int(A)
end
function sum_piece_depth(ctx::LayerCtx, rest::Int)
    m = Int(ctx.m)
    total = big(0)
    e = 1
    while e * m <= rest; e *= m; end
    r = rest
    while e >= 1
        total += (r ÷ e) * depth_bound(ctx, e)
        r %= e
        e ÷= m
    end
    return total
end

# --- elementary kernels -------------------------------------------------------

# One C kernel on bit `bit` of the flat index (0-based), over the whole array:
# (u, v) ↦ ((u+v) + i(u-v), (u+v) - i(u-v)) / 2, the halving deferred to an
# exact arithmetic shift on the guarded numerators.
function c_kernel!(A::Vector{GC{L}}, bit::Int) where {L}
    n = length(A)
    step = 1 << bit
    @inbounds for base in 0:2step:n-1
        for i in base+1:base+step
            u = A[i]; v = A[i+step]
            s = u + v
            dI = mul_i(u - v)
            A[i] = (s + dI) >> 1
            A[i+step] = (s - dI) >> 1
        end
    end
    return A
end
c_kernels!(A::Vector{GC{L}}, bits) where {L} = (for b in bits; c_kernel!(A, b); end; A)

# Diagonal sign (−1)^{popcount(idx & mask)} on every entry.
function sign_mask!(A::Vector{GC{L}}, mask::Int) where {L}
    mask == 0 && return A
    @inbounds for idx in 0:length(A)-1
        isodd(count_ones(idx & mask)) && (A[idx+1] = -A[idx+1])
    end
    return A
end
# Phase i^{popcount(idx & mask)} (the paper's S_D) on every entry.
function phase_mask!(A::Vector{GC{L}}, mask::Int) where {L}
    mask == 0 && return A
    @inbounds for idx in 0:length(A)-1
        A[idx+1] = mul_ipow(A[idx+1], count_ones(idx & mask))
    end
    return A
end
function scale_ipow!(A::Vector{GC{L}}, k::Int) where {L}
    k & 3 == 0 && return A
    @inbounds for i in eachindex(A)
        A[i] = mul_ipow(A[i], k)
    end
    return A
end

# --- the recursive C layer on a stream -----------------------------------------
#
# A stream holds R rows of 2^e entries: S[g 2^e + x + 1], where the e active
# bits of x are laid out slot-major: bit (h-1) f + j is x_{h,j}, the selected
# bit of chunk j in slot h (1 ≤ h ≤ m, 0 ≤ j < f).

# Apply C^{⊗e} to every row of the stream.
function c_group!(S::Vector{GC{L}}, R::Int, e::Int, ctx::LayerCtx) where {L}
    e == 0 && return S
    m = ctx.m
    if ctx.net === nothing || e < m || e < ctx.leaf || e % m != 0
        return c_kernels!(S, 0:e-1)
    end
    net = ctx.net
    W = net.W
    f = e ÷ Int(m)
    mint = Int(m)
    blk = 1 << e
    # pad the rows to a multiple of W with zero rows
    Rp = W * cld(R, W)
    if Rp > R
        resize!(S, Rp * blk)
        fill!(view(S, R*blk+1:Rp*blk), zero(GC{L}))
    end
    Rw = Rp ÷ W
    roles = [Vector{GC{L}}(undef, Rw * blk) for _ in 1:W]
    @inbounds for g in 0:Rp-1
        w = g % W + 1
        gw = g ÷ W
        copyto!(roles[w], gw * blk + 1, S, g * blk + 1, blk)
    end
    slotmask(h) = ((1 << f) - 1) << ((h - 1) * f)
    coordmask(coords) = isempty(coords) ? 0 : reduce(|, slotmask(h) for h in coords)
    # endpoint corrections before the network
    for w in 1:W
        sign_mask!(roles[w], coordmask(net.pre_z[w]))
    end
    cur = copy(net.src_label)
    touched = Int[]
    for g in net.gates
        empty!(touched)
        for u in g.updates
            push!(touched, u.target)
            for (r, _) in u.terms
                push!(touched, r)
            end
        end
        unique!(touched)
        for w in touched
            frame_change!(roles[w], Rw, e, f, cur[w], g.label, ctx)
            cur[w] = g.label
        end
        apply_gate!(roles, g)
    end
    for w in 1:W
        frame_change!(roles[w], Rw, e, f, cur[w], net.snk_label[w], ctx)
    end
    # endpoint corrections after the network, undo the signed exchange
    for w in 1:W
        sign_mask!(roles[w], coordmask(net.post_z[w]))
        scale_ipow!(roles[w], net.post_ipow[w] * f)
    end
    @inbounds for w in 1:W
        src = roles[net.rho[w]]
        sgn = net.route_sign[w]
        for g in 0:Rw-1
            gg = g * W + (w - 1)
            if sgn == 1
                copyto!(S, gg * blk + 1, src, g * blk + 1, blk)
            else
                for x in 1:blk
                    S[gg*blk+x] = -src[g*blk+x]
                end
            end
        end
    end
    if Rp > R
        @inbounds for i in R*blk+1:Rp*blk
            iszero(S[i]) || error("padded row not restored")
        end
        resize!(S, R * blk)
    end
    return S
end

# target += Σ coef·source pointwise over the role streams
function apply_gate!(roles::Vector{Vector{GC{L}}}, g::Gate) where {L}
    for u in g.updates
        tgt = roles[u.target]
        for (r, c) in u.terms
            src = roles[r]
            @inbounds for i in eachindex(tgt)
                tgt[i] = tgt[i] + apply_coef(c, src[i])
            end
        end
    end
    return roles
end
@inline function apply_coef(c::Coef, z::GC{L}) where {L}
    w = c.re == 1 ? z : c.re == -1 ? -z : c.re == 0 ? zero(GC{L}) : z * Int64(c.re)
    if c.im != 0
        w = w + (c.im == 1 ? mul_i(z) : c.im == -1 ? mul_negi(z) : mul_i(z) * Int64(c.im))
    end
    return c.sh == 0 ? w : w >> Int(c.sh)
end

# Orthonormal residual basis and kernel exponents of an edge, cached.
function edge_kernels(ctx::LayerCtx, from::Int, to::Int)
    get!(ctx.onb_cache, (from, to)) do
        net = ctx.net
        d, E = edge_residual(net, from, to)
        onb = f2orthonormal_basis(E)
        # i^{wt(v)[v·x]} with wt(v) odd: C_v for wt ≡ 1 (mod 4), C_v^{-1} for wt ≡ 3
        eps = [f2weight(v) % 4 == 1 ? 1 : -1 for v in onb]
        d < 0 && (eps .*= -1)      # a decreasing edge inverts the frame change
        (onb, eps)
    end
end

# Apply C_V C_U^{-1} (tensored over the f columns) to a role stream.
function frame_change!(S::Vector{GC{L}}, R::Int, e::Int, f::Int, from::Int, to::Int, ctx::LayerCtx) where {L}
    from == to && return S
    onb, eps = edge_kernels(ctx, from, to)
    isempty(onb) && return S
    m = Int(ctx.m)
    ρ = length(onb)
    cols = f2complete_basis(m, onb)            # M, columns v_1..v_ρ first
    icols = f2invert(cols)
    perm_M = column_perm(cols, m, f)           # x ↦ M x per column
    perm_Mi = column_perm(icols, m, f)
    blk = 1 << e
    tmp = similar(S)
    # g = P_M^{-1} S : g[x] = S[M x]
    @inbounds for g in 0:R-1, x in 0:blk-1
        tmp[g*blk+x+1] = S[g*blk+perm_M[x+1]+1]
    end
    for h in 1:ρ
        child_slot!(tmp, R, e, f, h, eps[h], ctx)
    end
    # S = P_M g : S[x] = g[M^{-1} x]
    @inbounds for g in 0:R-1, x in 0:blk-1
        S[g*blk+x+1] = tmp[g*blk+perm_Mi[x+1]+1]
    end
    return S
end

# Permutation table of x ∈ [2^(m f)] under a column-wise linear map given by
# its columns over F_2^m.
function column_perm(cols::Vector{F2Vec}, m::Int, f::Int)
    e = m * f
    perm = Vector{Int}(undef, 1 << e)
    # image of each single bit x_{h,j}: column h of the matrix, placed in column j
    img = zeros(Int, e)
    for h in 1:m, j in 0:f-1
        y = 0
        for h2 in 1:m
            cols[h][h2] && (y |= 1 << ((h2 - 1) * f + j))
        end
        img[(h-1)*f+j+1] = y
    end
    @inbounds for x in 0:(1<<e)-1
        y = 0
        xx = x
        while xx != 0
            b = trailing_zeros(xx)
            y ⊻= img[b+1]
            xx &= xx - 1
        end
        perm[x+1] = y
    end
    return perm
end

# One child: (C^{±1})^{⊗f} on the f bits of slot h of every row.  A negative
# kernel is one forward child wrapped in Z^{⊗f} on both sides and the phase
# (-i)^f: (C^{-1})^{⊗f} = (-i)^f Z^{⊗f} C^{⊗f} Z^{⊗f}.
function child_slot!(S::Vector{GC{L}}, R::Int, e::Int, f::Int, h::Int, eps::Int, ctx::LayerCtx) where {L}
    blk = 1 << e
    off = (h - 1) * f
    smask = ((1 << f) - 1) << off
    if eps == -1
        @inbounds for g in 0:R-1, x in 0:blk-1
            isodd(count_ones(x & smask)) && (S[g*blk+x+1] = -S[g*blk+x+1])
        end
    end
    # gather the slot's f bits as the active bits of a child stream whose rows
    # are (g, the other e-f bits)
    other = e - f
    lomask = (1 << off) - 1
    child = Vector{GC{L}}(undef, R * blk)
    cblk = 1 << f
    @inbounds for g in 0:R-1, y in 0:(1<<other)-1
        ylo = y & lomask
        yhi = y >> off
        base = (yhi << (off + f)) | ylo
        row = g * (1 << other) + y
        for z in 0:cblk-1
            child[row*cblk+z+1] = S[g*blk+(base|(z<<off))+1]
        end
    end
    c_group!(child, R * (1 << other), f, ctx)
    @inbounds for g in 0:R-1, y in 0:(1<<other)-1
        ylo = y & lomask
        yhi = y >> off
        base = (yhi << (off + f)) | ylo
        row = g * (1 << other) + y
        for z in 0:cblk-1
            S[g*blk+(base|(z<<off))+1] = child[row*cblk+z+1]
        end
    end
    if eps == -1
        @inbounds for g in 0:R-1, x in 0:blk-1
            isodd(count_ones(x & smask)) && (S[g*blk+x+1] = -S[g*blk+x+1])
        end
        scale_ipow!(S, -f)
    end
    return S
end

# --- the layer -----------------------------------------------------------------

# C^{⊗D} on the selected bit positions of a flat array (length a power of
# two), following the row construction of the paper: the first q_0 selected
# bits individually, then the rest in base-m pieces through the network.
function c_layer!(A::Vector{GC{L}}, selbits::Vector{Int}, ctx::LayerCtx) where {L}
    D = length(selbits)
    q0 = q0_bits(ctx)
    if ctx.net === nothing || D <= q0
        return c_kernels!(A, selbits)
    end
    c_kernels!(A, selbits[1:q0])
    rest = selbits[q0+1:end]
    m = Int(ctx.m)
    e = 1
    while e * m <= length(rest); e *= m; end
    pos = 1
    while e >= 1
        while length(rest) - pos + 1 >= e
            piece = rest[pos:pos+e-1]
            group_through_stream!(A, piece, ctx)
            pos += e
        end
        e ÷= m
    end
    return A
end

# Gather the entries of A into a stream whose active bits are `piece`, run
# the C group, and scatter back.
function group_through_stream!(A::Vector{GC{L}}, piece::Vector{Int}, ctx::LayerCtx) where {L}
    nb = trailing_zeros(length(A))
    e = length(piece)
    others = setdiff(0:nb-1, piece)
    R = 1 << length(others)
    blk = 1 << e
    S = Vector{GC{L}}(undef, R * blk)
    idx_of = (g, x) -> begin
        idx = 0
        for (k, b) in enumerate(others)
            (g >> (k - 1)) & 1 == 1 && (idx |= 1 << b)
        end
        for (k, b) in enumerate(piece)
            (x >> (k - 1)) & 1 == 1 && (idx |= 1 << b)
        end
        idx
    end
    @inbounds for g in 0:R-1, x in 0:blk-1
        S[g*blk+x+1] = A[idx_of(g, x)+1]
    end
    c_group!(S, R, e, ctx)
    @inbounds for g in 0:R-1, x in 0:blk-1
        A[idx_of(g, x)+1] = S[g*blk+x+1]
    end
    return A
end

# The simultaneous normalized butterfly layer: returns Q_p(H_0^{⊗D} A) on the
# selected bits, computed exactly on guarded numerators and truncated once
# toward zero (paper, Proposition on the simultaneous layer).
function butterfly_layer!(A::Vector{GC{L}}, selbits::Vector{Int}, ctx::LayerCtx) where {L}
    D = length(selbits)
    D == 0 && return A
    ispow2(length(A)) || throw(ArgumentError("array length must be a power of two"))
    G = guard_bits(ctx, D)
    Lw = cld(ctx.p + 2G + 4, 64)
    return _butterfly_layer!(Val(Lw), A, selbits, ctx, G)
end

function _butterfly_layer!(::Val{Lw}, A::Vector{GC{L}}, selbits::Vector{Int}, ctx::LayerCtx, G::Int) where {Lw,L}
    D = length(selbits)
    B = Vector{GC{Lw}}(undef, length(A))
    @inbounds for i in eachindex(A)
        B[i] = resize(GC{Lw}, A[i]) << G
    end
    mask = reduce(|, 1 << b for b in selbits)
    phase_mask!(B, mask)                       # S_D
    c_layer!(B, selbits, ctx)                  # C^{⊗D}
    phase_mask!(B, mask)                       # S_D
    # b^D = (-i/2)^{⌊D/2⌋} b^{D mod 2}, with b z = (z - i z)/2
    t = D ÷ 2
    scale_ipow!(B, -t)
    @inbounds for i in eachindex(B)
        z = B[i] >> t
        isodd(D) && (z = (z - mul_i(z)) >> 1)
        A[i] = resize(GC{L}, trunc_shr(z, G))
    end
    return A
end
