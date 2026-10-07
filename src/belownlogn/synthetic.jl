# Synthetic transforms over R_r = C[y]/(y^r + 1) and exact normalized
# polynomial products (paper, Sections 6 and 7).
#
# An array of M records in R_r is a flat Vector{GC{L}} of M r grid
# coefficients: record j occupies [j r + 1, (j+1) r], and the record index
# j runs over the d-1 ring axes of lengths t_i ∈ {r/2, r} in lexicographic
# order, axis 1 most significant.  Since every length is a power of two, the
# flat coefficient index is a bit concatenation: the polynomial index k takes
# the low ℓ = log2 r bits and axis i takes a_i = log2 t_i bits above those of
# axis i+1.
#
# ω_t = y^(2r/t) is a t-th root of unity in R_r, and multiplying by a power of
# y is a signed coefficient rotation.  The decimation-in-frequency recursion
# writes the forward transform B F_t^- in bit-reversed frequency order, the
# opposite procedure consumes that order and computes F_t^+ B^-1, and the
# rounds of all axes at a common named position h are grouped into one
# parallel butterfly layer (layers.jl) followed by one record-wise monomial.

struct SynthPlan
    r::Int
    ell::Int
    axes::Vector{Int}       # a_i = log2 t_i, i = 1..D (D = d-1 ring axes)
    offs::Vector{Int}       # flat bit offset of axis i
    M::Int                  # ∏ t_i records
    mM::Int                 # log2 M
    p::Int
    ctx::LayerCtx
    BL::Int                 # packing block width in limbs (≥ 4p bits)
    L3::Int                 # limbs holding a 3p+4-bit extracted coefficient
end

function SynthPlan(r::Int, axes::Vector{Int}, p::Int, ctx::LayerCtx)
    ispow2(r) && r >= 2 || throw(ArgumentError("r must be a power of two ≥ 2"))
    ell = trailing_zeros(r)
    all(a -> a == ell || a == ell - 1, axes) || throw(ArgumentError("axis lengths must be r/2 or r"))
    D = length(axes)
    offs = zeros(Int, D)
    off = ell
    for i in D:-1:1
        offs[i] = off
        off += axes[i]
    end
    mM = sum(axes; init=0)
    BL = cld(4p, 64)
    L3 = cld(3p + 4, 64)
    SynthPlan(r, ell, axes, offs, 1 << mM, mM, p, ctx, BL, L3)
end
nrecords(pl::SynthPlan) = pl.M
ncoeffs(pl::SynthPlan) = pl.M * pl.r

# --- monomials -------------------------------------------------------------------

# record ← y^E · record modulo y^r + 1, in place with a scratch of r coefficients
function rotate_record!(A::Vector{GC{L}}, base::Int, r::Int, E::Int, tmp::Vector{GC{L}}) where {L}
    E = mod(E, 2r)
    E == 0 && return A
    e, s = divrem(E, r)
    neg = isodd(e)
    @inbounds for k in 0:r-1
        tmp[k+1] = A[base+k+1]
    end
    @inbounds for k in 0:r-1
        t = k + s
        z = tmp[k+1]
        if t >= r
            t -= r
            z = -z
        end
        neg && (z = -z)
        A[base+t+1] = z
    end
    return A
end

# The grouped twiddle of a round at named position h on the participating
# axes: E = -Σ_i (2r/u_i) b_i k_i (mod 2r), u_i = 2^(h+1), b_i the branch bit
# written at position h, k_i the value of the lower h bits of axis i.
function twiddle_round!(A::Vector{GC{L}}, pl::SynthPlan, h::Int, part::Vector{Int}, dir::Int, tmp::Vector{GC{L}}) where {L}
    r = pl.r
    scale = r >> h                     # 2r / 2^(h+1)
    lowmask = (1 << h) - 1
    @inbounds for j in 0:pl.M-1
        E = 0
        for i in part
            ji = (j >> (pl.offs[i] - pl.ell)) & ((1 << pl.axes[i]) - 1)
            b = (ji >> h) & 1
            b == 0 && continue
            E -= scale * (ji & lowmask)
        end
        E == 0 && continue
        rotate_record!(A, j * r, r, dir * E, tmp)
    end
    return A
end

participating(pl::SynthPlan, h::Int) = [i for i in 1:length(pl.axes) if pl.axes[i] - 1 >= h]

# Forward synthetic transform, exact counterpart B F_t^-, error < √2 ℓ 2^-p.
function synth_forward!(A::Vector{GC{L}}, pl::SynthPlan) where {L}
    length(A) == ncoeffs(pl) || throw(ArgumentError("array size mismatch"))
    tmp = Vector{GC{L}}(undef, pl.r)
    for h in pl.ell-1:-1:0
        part = participating(pl, h)
        isempty(part) && continue
        selbits = [pl.offs[i] + h for i in part]
        butterfly_layer!(A, selbits, pl.ctx)
        twiddle_round!(A, pl, h, part, +1, tmp)
    end
    return A
end

# Opposite transform consuming the stored order, exact counterpart F_t^+ B^-1.
function synth_opposite!(A::Vector{GC{L}}, pl::SynthPlan) where {L}
    length(A) == ncoeffs(pl) || throw(ArgumentError("array size mismatch"))
    tmp = Vector{GC{L}}(undef, pl.r)
    for h in 0:pl.ell-1
        part = participating(pl, h)
        isempty(part) && continue
        twiddle_round!(A, pl, h, part, -1, tmp)
        selbits = [pl.offs[i] + h for i in part]
        butterfly_layer!(A, selbits, pl.ctx)
    end
    return A
end

# --- signed packing and the normalized ring product ----------------------------------
#
# Lemma (signed packing): with B_0 ≥ 2^4p, pack the four integer coefficient
# streams a, b, c, e of f = 2^-p (a + i b), g = 2^-p (c + i e) as signed
# evaluations u(B_0), multiply the four pairs with the established integer
# multiplier, recover the product coefficients by centered residues, reduce
# modulo y^r + 1 and truncate (fg / r) to the grid.  B_0 is 2^4p rounded up
# to a limb boundary, which only widens the separation.

struct RingWork{L3}
    packs::Vector{Memory{Limb}}
    prods::Vector{Memory{Limb}}
    re::Vector{FW{L3}}
    im::Vector{FW{L3}}
end
function RingWork(pl::SynthPlan)
    n = pl.r * pl.BL
    packs = [Memory{Limb}(undef, n) for _ in 1:4]
    prods = [Memory{Limb}(undef, 2n) for _ in 1:4]
    L3 = pl.L3
    RingWork{L3}(packs, prods, Vector{FW{L3}}(undef, 2pl.r), Vector{FW{L3}}(undef, 2pl.r))
end

# Pack the component (re or im) of record `base` as the two's complement
# evaluation at B_0; returns (magnitude length in limbs, negative?).  The
# borrow γ_j ∈ {-1, 0} carries the sign of the previous block.
function pack_component!(buf::Memory{Limb}, A::Vector{GC{L}}, base::Int, r::Int, BL::Int, useim::Bool) where {L}
    γ = zero(FW{L})
    minus1 = FW{L}(-1)
    @inbounds for j in 0:r-1
        z = A[base+j+1]
        u = useim ? z.im : z.re
        v = u + γ
        # the block is the sign extension of v to BL limbs: v itself when
        # v ≥ 0, B_0 + v when v < 0
        store_fw!(buf, j * BL, BL, v)
        γ = signbit(v) ? minus1 : zero(FW{L})
    end
    n = r * BL
    neg = !iszero(γ)
    neg && negate_twos!(buf, n)
    return normlen(buf, 0, n), neg
end

# Z = ±|A|·|C| as a two's complement integer over 2 r BL limbs in `out`.
function signed_product!(out::Memory{Limb}, a::Memory{Limb}, la::Int, nega::Bool,
                         c::Memory{Limb}, lc::Int, negc::Bool, nlimbs::Int)
    fill!(out, zero(Limb))
    (la == 0 || lc == 0) && return out
    if la >= lc
        mul!(out, 0, a, 0, la, c, 0, lc)
    else
        mul!(out, 0, c, 0, lc, a, 0, la)
    end
    (nega ⊻ negc) && negate_twos!(out, nlimbs)
    return out
end

# Centered extraction of the 2r product coefficients of a packed product.
function extract_coeffs!(h::Vector{FW{L3}}, Z::Memory{Limb}, r::Int, BL::Int) where {L3}
    ρ = zero(FW{L3})
    one3 = FW{L3}(1)
    @inbounds for j in 0:2r-1
        off = j * BL
        top = Z[off+BL] >> 63 == 1
        # low L3 limbs with the block's own sign: an all-ones top means q_j is
        # B_0 minus a small number, i.e. the signed reading of the low limbs
        ext = top ? ALLONES : UInt64(0)
        q = FW{L3}(ntuple(i -> i <= BL ? Z[off+i] : ext, Val(L3)))
        h[j+1] = q + ρ
        ρ = top ? one3 : zero(FW{L3})
    end
    return h
end

# out record ← Q_p(f g / r) for records f, g (grid polynomials of norm ≤ 1).
function ring_product!(out::Vector{GC{L}}, obase::Int, f::Vector{GC{L}}, fbase::Int,
                       g::Vector{GC{L}}, gbase::Int, pl::SynthPlan, wk::RingWork{L3}) where {L,L3}
    r, BL, p = pl.r, pl.BL, pl.p
    n = r * BL
    la, nega = pack_component!(wk.packs[1], f, fbase, r, BL, false)   # a = Re f
    lb, negb = pack_component!(wk.packs[2], f, fbase, r, BL, true)    # b = Im f
    lc, negc = pack_component!(wk.packs[3], g, gbase, r, BL, false)   # c = Re g
    le, nege = pack_component!(wk.packs[4], g, gbase, r, BL, true)    # e = Im g
    signed_product!(wk.prods[1], wk.packs[1], la, nega, wk.packs[3], lc, negc, 2n)  # ac
    signed_product!(wk.prods[2], wk.packs[2], lb, negb, wk.packs[4], le, nege, 2n)  # be
    signed_product!(wk.prods[3], wk.packs[1], la, nega, wk.packs[4], le, nege, 2n)  # ae
    signed_product!(wk.prods[4], wk.packs[2], lb, negb, wk.packs[3], lc, negc, 2n)  # bc
    ac = extract_coeffs!(wk.re, wk.prods[1], r, BL)
    be = extract_coeffs!(wk.im, wk.prods[2], r, BL)
    # real numerators ac - be, wrapped negacyclically, then truncated by 2^(p+ℓ)
    sh = p + pl.ell
    @inbounds for k in 0:r-1
        hi = k + r
        H = (ac[k+1] - be[k+1]) - (ac[hi+1] - be[hi+1])
        out[obase+k+1] = GC{L}(resize(FW{L}, trunc_shr(H, sh)), zero(FW{L}))
    end
    ae = extract_coeffs!(wk.re, wk.prods[3], r, BL)
    bc = extract_coeffs!(wk.im, wk.prods[4], r, BL)
    @inbounds for k in 0:r-1
        hi = k + r
        H = (ae[k+1] + bc[k+1]) - (ae[hi+1] + bc[hi+1])
        z = out[obase+k+1]
        out[obase+k+1] = GC{L}(z.re, resize(FW{L}, trunc_shr(H, sh)))
    end
    return out
end

# --- the normalized convolution interface ------------------------------------------

# M_r(f, g) = (f * g) / (r M) on disk grid arrays; `gt` may be supplied as the
# already transformed second operand (the chirp kernel is reused across the
# transforms of one multiplication).  Returns the result in `f`'s storage.
function synth_convolution!(f::Vector{GC{L}}, g::Vector{GC{L}}, pl::SynthPlan;
                            g_transformed::Bool=false) where {L}
    synth_forward!(f, pl)
    gt = g_transformed ? g : synth_forward!(copy(g), pl)
    wk = RingWork(pl)
    r = pl.r
    for j in 0:pl.M-1
        ring_product!(f, j * r, f, j * r, gt, j * r, pl, wk)
    end
    synth_opposite!(f, pl)
    # exact scale by M = 2^mM
    @inbounds for i in eachindex(f)
        f[i] = f[i] << pl.mM
    end
    return f
end
