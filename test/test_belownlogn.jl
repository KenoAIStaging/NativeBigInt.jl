# Tests for the sub-n-log-n multiplier (src/belownlogn/), layer by layer:
# exact arithmetic, the finite networks and their labels, the frame engine
# and simultaneous butterfly layer, synthetic transforms and ring products,
# Gaussian resampling, and the end-to-end product against Base.BigInt.

const BN = NativeBigInt.BelowNLogN
using .BN: FW, GC, tobig, mul_wide, mul_trunc, trunc_shr, round_shr, magbits, fits, resize, mul_i, mul_ipow,
    F2Space, f2zero_space, f2line, f2unit, f2full_space, f2span, f2dot, f2weight, f2project, f2perp_in,
    f2orthonormal_basis, f2nondegenerate,
    motif_counts, motif_deficit, motif_network, complex_motif_network, run_scalar!, network_edges,
    edge_residual, residual_rank_sum,
    LinearNetwork, Gate, Update, COEF_ONE, COEF_MINUS, LayerCtx, paper_layer_ctx, q0_bits, guard_bits,
    frame_change!, c_group!, c_kernels!, butterfly_layer!,
    SynthPlan, synth_forward!, synth_convolution!, ring_product!, RingWork, rotate_record!,
    AxisMaps, Box, ChirpData, source_transform!, source_transform_plus!,
    Params, MulPlan, multiply!, mul_belownlogn, mul_limbs!

const CQ = Complex{Rational{BigInt}}

@testset "fixed-width Gaussian dyadics" begin
    rng = MersenneTwister(1)
    for L in (1, 2, 3, 5)
        W = FW{L}
        rb(bits) = (x = rand(rng, big(0):(big(2)^bits-1)); rand(rng, Bool) ? -x : x)
        for _ in 1:100
            a = rb(64L - 2); b = rb(64L - 2)
            A, B = W(a), W(b)
            @test BigInt(A) == a && BigInt(B) == b
            @test BigInt(A + B) == a + b
            @test BigInt(A - B) == a - b
            @test BigInt(-A) == -a
            k = rand(rng, 0:64L-1)
            @test BigInt(A >> k) == a >> k
            @test BigInt(trunc_shr(A, k)) == div(a, big(2)^k)
            @test k == 0 || BigInt(round_shr(A, k)) == fld(a + big(2)^k ÷ 2, big(2)^k)
            s = rand(rng, 0:30)
            @test BigInt(W(a >> s) << s) == (a >> s) << s
            @test cmp(A, B) == cmp(a, b)
            @test BigInt(mul_wide(FW{2L + 1}, A, B)) == a * b
            @test magbits(A) == (a == 0 ? 0 : ndigits(abs(a), base=2))
            @test fits(A, 64L - 1)
            @test BigInt(resize(FW{L + 2}, A)) == a
        end
        p = 64L - 40
        for _ in 1:30
            za = GC{L}(rb(p), rb(p)); zb = GC{L}(rb(p), rb(p))
            (ar, ai) = tobig(za); (br, bi) = tobig(zb)
            w = mul_trunc(za, zb, p, GC{2L + 1})
            @test tobig(w) == (div(ar * br - ai * bi, big(2)^p), div(ar * bi + ai * br, big(2)^p))
            @test tobig(mul_i(za)) == (-ai, ar)
            @test tobig(mul_ipow(za, 3)) == (ai, -ar)
        end
    end
end

@testset "finite networks: counts and the exchange lemma" begin
    # the paper's constants at h = 100
    cb = motif_counts(100, :bit); cc = motif_counts(100, :complex)
    @test cb.v == 161700 && cb.N == 4227952113000000 && cb.I == 78440670000 && cb.m == 1000000
    @test cb.z == 13968 && cc.z == 147731
    @test cb.W == big"177176569091445000000"
    @test cc.W == big"1873807244643542670000"
    @test cb.s == big"177176569088785861287000000"
    @test cc.s == big"1873807244636671267308000000"
    @test motif_deficit(cb) == 339 // 22587335000000
    @test motif_deficit(cc) == 73 // 19906842167500
    @test cb.L // cb.N == 100 // 539 && cc.L // cc.N == 101 // 539
    # the smallest instance whose complex deficit is positive
    @test motif_deficit(motif_counts(21, :complex)) <= 0
    @test motif_deficit(motif_counts(22, :complex)) > 0
    # (X_a, Y_a) ↦ (-Y_a, X_a), scratch restored, for both constructions
    rng = MersenneTwister(7)
    for h in 3:4, kind in (:bit, :complex)
        net = motif_network(h, kind; labels=false)
        cnt = motif_counts(h, kind)
        @test net.W == cnt.W
        N = Int(cnt.N)
        if kind == :bit
            vals = rand(rng, Bool, net.W); orig = copy(vals)
            run_scalar!(vals, net)
            @test vals[1:N] == orig[N+1:2N] && vals[N+1:2N] == orig[1:N]
        else
            R = Complex{Rational{Int128}}
            vals = [R(rand(rng, -5:5) // rand(rng, 1:4), rand(rng, -5:5) // rand(rng, 1:4)) for _ in 1:net.W]
            orig = copy(vals)
            run_scalar!(vals, net)
            @test vals[1:N] == -orig[N+1:2N] && vals[N+1:2N] == orig[1:N]
        end
        @test vals[2N+1:end] == orig[2N+1:end]
    end
end

@testset "finite networks: labels and residuals" begin
    for h in 3:4
        net = complex_motif_network(h)
        cnt = motif_counts(h, :complex)
        total = 0
        dec = Int[]
        for (_, from, to) in network_edges(net)
            from == to && continue
            d, E = edge_residual(net, from, to)
            total += abs(d)
            d < 0 && push!(dec, d)
            if BN.f2dim(E) > 0 && h >= 4
                # orthonormal residual bases exist once a coordinate outside a
                # triple's support is available (h ≥ 4); at h = 3 the residual
                # P ⊗ t^⟂ is alternating
                ob = f2orthonormal_basis(E)
                @test length(ob) == BN.f2dim(E)
                @test all(f2dot(x, x) for x in ob)
                @test all(!f2dot(ob[i], ob[j]) for i in 1:length(ob) for j in i+1:length(ob))
                @test f2span(E.n, ob) == E
            end
        end
        @test total == cnt.s                               # Σ|Δdim| = W m - 2N + 2L
        @test all(==(-h), dec) && length(dec) == cnt.I * cnt.c  # only the central returns decrease
        @test all(f2nondegenerate, net.labels)
    end
end

# --- the array engine ---------------------------------------------------------------------------

# explicit frame C_U = 2^-m Hd^{⊗m} diag(i^{wt(P_U x)}) Hd^{⊗m}
function bn_hadamard(m)
    H = fill(CQ(1), 1, 1)
    for _ in 1:m
        H = [H H; H -H]
    end
    H
end
bn_bits(x, m) = BitVector([(x >> (k - 1)) & 1 == 1 for k in 1:m])
function bn_frame_matrix(U::F2Space)
    m = U.n; n = 1 << m
    H = bn_hadamard(m)
    D = zeros(CQ, n, n)
    for x in 0:n-1
        D[x+1, x+1] = CQ(im)^(f2weight(f2project(U, bn_bits(x, m))) % 4)
    end
    (H * D * H) .// (big(2)^m)
end
# O^{⊗f} on f columns in the engine's slot-major bit layout
function bn_column_tensor(O, m, f)
    n = 1 << (m * f)
    R = zeros(CQ, n, n)
    col(x, j) = (c = 0; for h in 1:m; ((x >> ((h - 1) * f + j)) & 1 == 1) && (c |= 1 << (h - 1)); end; c)
    for x in 0:n-1, y in 0:n-1
        v = CQ(1)
        for j in 0:f-1
            v *= O[col(x, j)+1, col(y, j)+1]
            iszero(v) && break
        end
        R[x+1, y+1] = v
    end
    R
end
bn_toq(z::GC, G) = CQ(BigInt(z.re) // big(2)^G, BigInt(z.im) // big(2)^G)

# A three-role miniature with the paper's label pattern: X: ⟨e1⟩ → F,
# Y: 0 → ⟨e1⟩^⟂, scratch Z: 0 → F, labels rising and falling along the
# wires so that inverse children and the endpoint corrections are exercised.
function bn_mini_network()
    m = 2
    L0 = f2zero_space(m); Le1 = f2line(f2unit(m, 1)); Le2 = f2line(f2unit(m, 2)); LF = f2full_space(m)
    X, Y, Z = 1, 2, 3
    g(lab, t, s, c) = Gate(lab, [Update(t, [(s, c)])])
    P, M = COEF_ONE, COEF_MINUS
    gates = [g(1, Y, Z, M), g(2, Z, X, P), g(4, Y, Z, P), g(4, Z, X, M),      # Y += X
             g(4, Z, Y, P), g(2, X, Z, M), g(2, Z, Y, M), g(4, X, Z, P),      # X -= Y
             g(4, Y, Z, M), g(4, Z, X, P), g(3, Y, Z, P), g(4, Z, X, M)]      # Y += X
    LinearNetwork(:complex, 3, m, [L0, Le1, Le2, LF], [2, 1, 1], [4, 3, 4], gates, [2, 1, 3], [1, -1, 1],
                  [[1], Int[], Int[]], [Int[], [1], Int[]], [0, 1, 0], ["X", "Y", "Z"])
end

@testset "frame engine and simultaneous layer" begin
    net = bn_mini_network()
    R = Complex{Rational{Int128}}
    vals = [R(1 // 2, 3), R(-2, 1 // 4), R(5 // 3, -7)]
    run_scalar!(vals, net)
    @test vals == [R(2, -1 // 4), R(1 // 2, 3), R(5 // 3, -7)]
    rng = MersenneTwister(3)
    Lt = 2
    ctx = LayerCtx(net, 4, 40; leaf=1)
    # every edge type: C_V C_U^-1 tensored over f columns
    for (from, to) in [(1, 2), (2, 1), (1, 4), (4, 1), (2, 4), (4, 2), (3, 4), (4, 3), (1, 3), (3, 1)], f in 1:2
        m = 2; e = m * f; G = 2e + 2
        O = bn_column_tensor(bn_frame_matrix(net.labels[to]) * inv(bn_frame_matrix(net.labels[from])), m, f)
        Rr = 3; blk = 1 << e
        S = [GC{Lt}(FW{Lt}(rand(rng, -50:50)) << G, FW{Lt}(rand(rng, -50:50)) << G) for _ in 1:Rr*blk]
        ref = [bn_toq(z, G) for z in S]
        frame_change!(S, Rr, e, f, from, to, ctx)
        for g in 0:Rr-1
            @test O * ref[g*blk+1:(g+1)*blk] == [bn_toq(z, G) for z in S[g*blk+1:(g+1)*blk]]
        end
    end
    # the whole network: C^{⊗e} on every role, rows not a multiple of W
    for f in 1:2
        e = 2f; G = 40; Rr = 5
        S = [GC{Lt}(FW{Lt}(rand(rng, -50:50)) << G, FW{Lt}(rand(rng, -50:50)) << G) for _ in 1:Rr*(1<<e)]
        ref = copy(S); c_kernels!(ref, 0:e-1)
        c_group!(S, Rr, e, ctx)
        @test S == ref
    end
    # the normalized layer against exact H_0^{⊗D} with one truncation
    function h0_exact(A::Vector{CQ}, selbits)
        B = copy(A)
        for b in selbits
            step = 1 << b
            for base in 0:2step:length(B)-1, i in base+1:base+step
                u, v = B[i], B[i+step]
                B[i] = (u + v) // 2; B[i+step] = (u - v) // 2
            end
        end
        B
    end
    truncq(q::Rational, p) = div(numerator(q) * big(2)^p, denominator(q))
    p = 40
    for (ctx2, nb, sel) in [(LayerCtx(net, 8, p; leaf=1), 12, collect(0:9)),          # one network level, f = 1
                             (LayerCtx(net, 8, p; leaf=2), 14, [0, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 1]),  # f = 2 at the top
                             (paper_layer_ctx(8, p), 10, [1, 3, 4, 6, 8])]            # the paper's constants: individual kernels
        A = [GC{Lt}(FW{Lt}(rand(rng, -(big(2)^p):big(2)^p) >> 1), FW{Lt}(rand(rng, -(big(2)^p):big(2)^p) >> 1)) for _ in 1:(1<<nb)]
        Aq = [CQ(BigInt(z.re) // big(2)^p, BigInt(z.im) // big(2)^p) for z in A]
        butterfly_layer!(A, sel, ctx2)
        exact = h0_exact(Aq, sel)
        @test all(tobig(A[i]) == (truncq(real(exact[i]), p), truncq(imag(exact[i]), p)) for i in eachindex(A))
    end
    @test q0_bits(paper_layer_ctx(8, p)) == 71
    @test q0_bits(paper_layer_ctx(1000, p)) == 71
end

# --- synthetic transforms ----------------------------------------------------------------------

function bn_polymul_neg(f::Vector{CQ}, g::Vector{CQ})
    r = length(f); h = zeros(CQ, r)
    for i in 1:r, j in 1:r
        k = i + j - 2
        k < r ? (h[k+1] += f[i] * g[j]) : (h[k-r+1] -= f[i] * g[j])
    end
    h
end
function bn_ypow(f::Vector{CQ}, E)
    r = length(f); E = mod(E, 2r); e, s = divrem(E, r); h = zeros(CQ, r)
    for k in 0:r-1
        t = k + s; z = f[k+1]
        t >= r && (t -= r; z = -z)
        isodd(e) && (z = -z)
        h[t+1] = z
    end
    h
end

@testset "synthetic transforms and ring products" begin
    rng = MersenneTwister(11)
    p = 40; L = 2
    randgrid() = GC{L}(FW{L}(rand(rng, -(big(2)^p):big(2)^p) >> 1), FW{L}(rand(rng, -(big(2)^p):big(2)^p) >> 1))
    toq(z) = CQ(BigInt(z.re) // big(2)^p, BigInt(z.im) // big(2)^p)
    truncq(q::Rational) = div(numerator(q) * big(2)^p, denominator(q))
    ctx = paper_layer_ctx(4, p)
    for _ in 1:10
        r = 8
        A = [randgrid() for _ in 1:r]; E = rand(rng, -40:40)
        ref = bn_ypow([toq(z) for z in A], E)
        rotate_record!(A, 0, r, E, similar(A))
        @test [toq(z) for z in A] == ref
    end
    for r in (2, 4, 16)
        pl = SynthPlan(r, Int[], p, ctx); wk = RingWork(pl)
        for _ in 1:5
            f = [randgrid() for _ in 1:r]; g = [randgrid() for _ in 1:r]; out = similar(f)
            ring_product!(out, 0, f, 0, g, 0, pl, wk)
            ref = bn_polymul_neg([toq(z) for z in f], [toq(z) for z in g]) ./ r
            @test all(tobig(out[k]) == (truncq(real(ref[k])), truncq(imag(ref[k]))) for k in 1:r)
        end
    end
    # normalized convolution (f * g)/(r M) over the cyclic axes with negacyclic records
    for (r, axes) in [(4, [2]), (4, [1, 2]), (8, [3, 2]), (2, [1, 1])]
        pl = SynthPlan(r, axes, p, ctx)
        n = pl.M * r
        A = [randgrid() for _ in 1:n]; g = [randgrid() for _ in 1:n]
        Aq = [toq(z) for z in A]; gq = [toq(z) for z in g]
        D = length(axes); ts = [1 << a for a in axes]; strides = [prod(ts[i+1:end]; init=1) for i in 1:D]
        conv = zeros(CQ, n)
        for j1 in 0:pl.M-1, j2 in 0:pl.M-1
            j = 0
            for i in 1:D
                j += (((j1 ÷ strides[i]) % ts[i] + (j2 ÷ strides[i]) % ts[i]) % ts[i]) * strides[i]
            end
            conv[j*r+1:(j+1)*r] .+= bn_polymul_neg(Aq[j1*r+1:(j1+1)*r], gq[j2*r+1:(j2+1)*r])
        end
        conv ./= (r * pl.M)
        got = copy(A); synth_convolution!(got, g, pl)
        err = maximum(max(abs(real(toq(got[i]) - conv[i])), abs(imag(toq(got[i]) - conv[i]))) for i in 1:n)
        @test err < sqrt(2) * pl.M * (3pl.ell + 1) * 2.0^-p     # Proposition on synthetic convolution
    end
end

# --- resampling -----------------------------------------------------------------------------------

# (R F_s u)_j = (F_s u)_{t j mod s} along each axis, in BigFloat
function bn_direct_RF(u::Vector{Complex{BigFloat}}, t::Vector{Int}, s::Vector{Int}, sgn)
    d = length(t); strides = [prod(t[i+1:end]; init=1) for i in 1:d]
    out = copy(u)
    for i in 1:d
        new = zeros(Complex{BigFloat}, length(u))
        for idx in 0:length(u)-1
            ji = (idx ÷ strides[i]) % t[i]
            ji < s[i] || continue
            base = idx - ji * strides[i]
            freq = mod(t[i] * ji, s[i])
            acc = zero(Complex{BigFloat})
            for k in 0:s[i]-1
                acc += out[base+k*strides[i]+1] * exp(sgn * 2 * BigFloat(pi) * im * freq * k / s[i])
            end
            new[idx+1] = acc / s[i]
        end
        out = new
    end
    out
end

@testset "Gaussian resampling with retained permutations" begin
    rng = MersenneTwister(5)
    for (t, s) in [([32], [29]), ([8, 16], [7, 13])]
        d = length(t); b = 12; p = 160; L = 3
        α = ceil(Int, (12 * d^2 * b)^0.25); γ = 2d * α^2
        setprecision(BigFloat, p + 40) do
            ctx = paper_layer_ctx(d, p)
            pl = SynthPlan(t[end], [trailing_zeros(x) for x in t[1:end-1]], p, ctx)
            box = Box(t, s)
            axes = [AxisMaps{L}(s[i], t[i], p, α) for i in 1:d]
            cd = ChirpData{L}(box, pl, p)
            x = [zero(GC{L}) for _ in 1:box.size]
            for idx in 0:box.size-1
                all(((idx ÷ box.strides[i]) % t[i]) < s[i] for i in 1:d) || continue
                x[idx+1] = GC{L}(FW{L}(rand(rng, -(big(2)^(p-2)):big(2)^(p-2))), FW{L}(rand(rng, -(big(2)^(p-2)):big(2)^(p-2))))
            end
            toC(z) = Complex{BigFloat}(BigFloat(tobig(z)[1]) / big(2)^p, BigFloat(tobig(z)[2]) / big(2)^p)
            xq = toC.(x)
            T = prod(t); Lg = log2(T)
            Es = big(2)^γ * (2d * p^2 + 8T * Lg)                 # the paper's source error E_s
            for (sgn, f!) in ((-1, source_transform!), (+1, source_transform_plus!))
                ref = bn_direct_RF(xq, t, s, sgn)
                y = copy(x); f!(y, box, axes, cd, γ)
                err = maximum(abs(toC(y[i]) - ref[i]) for i in eachindex(y))
                @test err * big(2)^p < Es
            end
        end
    end
end

# --- the multiplication ------------------------------------------------------------------------------

@testset "multiplication below n log n" begin
    rng = MersenneTwister(2)
    # the paper's formulas give d = 1 for every representable length
    @test Params(2^40).d == 1
    @test Params(1000).p >= 6 * Params(1000).b
    for n in [1, 2, 3, 5, 8, 13, 31, 64, 100, 257, 1000], d in (nothing, 2, 3)
        x = rand(rng, big(0):big(2)^n-1); y = rand(rng, big(0):big(2)^n-1)
        z = mul_belownlogn(NBig(x), NBig(y); d)
        @test BigInt(z) == x * y
    end
    # extreme digit patterns and signs
    for n in (7, 64, 129, 2000)
        for (x, y) in ((big(2)^n - 1, big(2)^n - 1), (big(2)^(n-1), big(2)^n - 1), (big(1), big(2)^n - 1))
            @test BigInt(mul_belownlogn(NBig(x), NBig(y))) == x * y
            @test BigInt(mul_belownlogn(NBig(-x), NBig(y))) == -x * y
        end
    end
    @test iszero(mul_belownlogn(NBig(0), NBig(12345)))
    # unequal lengths through the mpn entry
    for _ in 1:20
        m = rand(rng, 1:6); n = rand(rng, 1:m)
        a = rand(rng, big(2)^(64m-1):big(2)^(64m)-1); b = rand(rng, big(2)^(64n-1):big(2)^(64n)-1)
        A, B = NBig(a), NBig(b)
        r = Memory{UInt64}(undef, m + n)
        mul_limbs!(r, 0, A.limbs, 0, m, B.limbs, 0, n)
        @test BigInt(NativeBigInt.nbig_from_limbs(1, r, m + n)) == a * b
    end
    # differential sweep over random lengths
    for _ in 1:60
        n = rand(rng, 1:1500)
        x = rand(rng, big(0):big(2)^n-1); y = rand(rng, big(0):big(2)^n-1)
        @test BigInt(mul_belownlogn(NBig(x), NBig(y))) == x * y
    end
end

@testset "dispatch at the paper's cutoff" begin
    using NativeBigInt: belownlogn_b, above_belownlogn_cutoff, MUL_BELOWNLOGN_THRESHOLD
    # b = ⌈log2(64m)⌉ for the m-limb input string
    @test belownlogn_b(1) == 6
    @test belownlogn_b(2) == 7
    @test belownlogn_b(3) == 8
    @test belownlogn_b(2^20) == 26
    @test belownlogn_b(2^20 + 1) == 27
    # the constant is the cutoff on the log2∘log6 scale: b = 6^(2^k) maps to k
    setprecision(BigFloat, 256) do
        for k in (0, 1, 2, 5, 10, 20, 40)
            b = BigFloat(6)^(BigFloat(2)^k)
            @test log2(log(big(6), b)) ≈ k atol = 1e-60
        end
    end
    @test MUL_BELOWNLOGN_THRESHOLD == 131    # K = ⌊d^c⌋ ≥ 6 ⟺ b ≥ 6^(2^131)
    # no representable operand reaches it (b ≤ 69 gives log2(log6 b) < 5)
    for m in (1, 2, 3, 64, 2^20, 2^40, 2^57, typemax(Int) >> 1, typemax(Int))
        @test !above_belownlogn_cutoff(m)
        @test log2(log(6.0, Float64(belownlogn_b(m)))) < 5
    end
end
