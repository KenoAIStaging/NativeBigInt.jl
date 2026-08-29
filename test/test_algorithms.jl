# Algorithm tests
using NativeBigInt: Limb, add_carry!, cmp_padded, abs_diff!, kar_scratch_len, MUL_KARATSUBA_THRESHOLD, divrem!,
    div_blocks!, DcEngine, MuEngine, invert_pi1, DC_DIV_THRESHOLD,
    invertappr!, invertappr_scratch_len, INV_NEWTON_THRESHOLD,
    mu_divrem!
using Random: MersenneTwister

amem(v::Vector{UInt64}) = (m = Memory{UInt64}(undef, length(v)); copyto!(m, v); m)
atoref(m, off, n) = (x = big(0); for i in n:-1:1; x = (x << 64) | m[off+i]; end; x)
afrombig(x::BigInt, n) = (v = zeros(UInt64, n); for i in 1:n; v[i] = UInt64(x & typemax(UInt64)); x >>= 64; end; amem(v))

@testset "kar helpers" begin
    # cmp_padded: value comparison with unnormalized (zero-padded) operands
    @test cmp_padded(amem([UInt64(5), UInt64(0)]), 0, 2, amem([UInt64(5)]), 0, 1) == 0
    @test cmp_padded(amem([UInt64(4), UInt64(1)]), 0, 2, amem([UInt64(9)]), 0, 1) == 1
    @test cmp_padded(amem([UInt64(4), UInt64(0)]), 0, 2, amem([UInt64(9)]), 0, 1) == -1
    @test cmp_padded(amem([UInt64(9), UInt64(2)]), 0, 2, amem([UInt64(9), UInt64(2)]), 0, 2) == 0

    # abs_diff!: x = [lo (2 limbs) | hi (1 limb)]
    d = Memory{UInt64}(undef, 2)
    # lo = 7, hi = 9 -> |7-9| = 2, negative
    @test abs_diff!(d, 0, amem([UInt64(7), UInt64(0), UInt64(9)]), 0, 2, 1) == true
    @test atoref(d, 0, 2) == 2
    # lo = B+9, hi = 7 -> lo-hi = B+2, positive
    @test abs_diff!(d, 0, amem([UInt64(9), UInt64(1), UInt64(7)]), 0, 2, 1) == false
    @test atoref(d, 0, 2) == (big(1) << 64) + 2
    # equal halves -> zero, non-negative
    @test abs_diff!(d, 0, amem([UInt64(3), UInt64(4), UInt64(3), UInt64(4)]), 0, 2, 2) == false
    @test atoref(d, 0, 2) == 0

    # add_carry!: ripple through a typemax limb
    r = amem([typemax(UInt64), UInt64(0)])
    add_carry!(r, 0, 2, 1, UInt64(1))
    @test atoref(r, 0, 2) == big(1) << 64
    # c == 0 is a no-op
    r2 = amem([UInt64(7)])
    add_carry!(r2, 0, 1, 1, UInt64(0))
    @test r2[1] == 7

    # kar_scratch_len
    @test kar_scratch_len(MUL_KARATSUBA_THRESHOLD - 1) == 0
    h2 = (MUL_KARATSUBA_THRESHOLD + 1) >> 1
    @test kar_scratch_len(MUL_KARATSUBA_THRESHOLD) == 4h2 + kar_scratch_len(h2)
    @test kar_scratch_len(4 * MUL_KARATSUBA_THRESHOLD) > kar_scratch_len(2 * MUL_KARATSUBA_THRESHOLD)
end

using NativeBigInt: mul_kar!

@testset "mul_kar! balanced" begin
    rng = MersenneTwister(123)
    T = MUL_KARATSUBA_THRESHOLD
    check(n, a, b) = begin
        r = Memory{UInt64}(undef, 2n)
        scratch = Memory{UInt64}(undef, kar_scratch_len(n))
        mul_kar!(r, 0, a, 0, b, 0, n, scratch, 0)
        @test atoref(r, 0, 2n) == atoref(a, 0, n) * atoref(b, 0, n)
    end
    # sizes spanning the threshold, odd/even splits, two+ recursion levels
    for n in (T - 1, T, T + 1, 2T, 2T + 1, 4T + 3), trial in 1:10
        check(n, amem(rand(rng, UInt64, n)), amem(rand(rng, UInt64, n)))
    end
    # adversarial: all-ones limbs (maximum carry chaining)
    for n in (T + 1, 2T + 1)
        a = amem(fill(typemax(UInt64), n))
        check(n, a, a)
    end
    # 2^k patterns: single high bit times single high bit minus one
    for n in (T + 1, 2T)
        a = afrombig(big(1) << (64n - 1), n)
        b = afrombig((big(1) << (64n - 1)) - 1, n)
        check(n, a, b)
    end
    # zero middle difference: a_lo == a_hi exactly
    n = 2 * (T + 1)
    half = rand(rng, UInt64, T + 1)
    check(n, amem([half; half]), amem(rand(rng, UInt64, n)))
end

using NativeBigInt: mul!

@testset "mul! general" begin
    rng = MersenneTwister(7)
    T = MUL_KARATSUBA_THRESHOLD
    checkmul(m, n, a, b) = begin
        r = Memory{UInt64}(undef, m + n)
        mul!(r, 0, a, 0, m, b, 0, n)
        @test atoref(r, 0, m + n) == atoref(a, 0, m) * atoref(b, 0, n)
    end
    # unbalanced shapes: basecase small-n, exact multiples, ragged tails,
    # tail chunk itself above/below threshold
    for (m, n) in ((100, 3), (64, 33), (2T, T), (2T + 5, T), (3T + 2, T + 1),
                   (200, T), (2T + T ÷ 2, T)), trial in 1:5
        checkmul(m, n, amem(rand(rng, UInt64, m)), amem(rand(rng, UInt64, n)))
    end
    # all-ones stress across the chunk boundaries
    m, n = 3T + 2, T + 1
    checkmul(m, n, amem(fill(typemax(UInt64), m)), amem(fill(typemax(UInt64), n)))
    # m == n delegates to balanced path
    for nn in (T, 2T + 1)
        checkmul(nn, nn, amem(rand(rng, UInt64, nn)), amem(rand(rng, UInt64, nn)))
    end
    # differential sweep vs BigInt over random sizes
    for trial in 1:60
        m = rand(rng, 1:3T); n = rand(rng, 1:m)
        checkmul(m, n, amem(rand(rng, UInt64, m)), amem(rand(rng, UInt64, n)))
    end
end

using NativeBigInt: MUL_FPNTT_THRESHOLD, SQR_FPNTT_THRESHOLD, sqr!

@testset "mul!/sqr! across the Karatsuba → NTT crossover" begin
    rng = MersenneTwister(41)
    T = MUL_FPNTT_THRESHOLD
    # balanced and unbalanced end-to-end sizes straddling the dispatch switch
    for (m, n) in ((T - 1, T - 1), (T, T), (T + 1, T + 1), (3T, T ÷ 2),
                   (2T + T ÷ 2, T + 1)),
        trial in 1:3
        a = amem(rand(rng, UInt64, m)); b = amem(rand(rng, UInt64, n))
        r = Memory{UInt64}(undef, m + n)
        mul!(r, 0, a, 0, m, b, 0, n)
        @test atoref(r, 0, m + n) == atoref(a, 0, m) * atoref(b, 0, n)
    end
    for n in (SQR_FPNTT_THRESHOLD - 1, SQR_FPNTT_THRESHOLD,
              2SQR_FPNTT_THRESHOLD + 1), trial in 1:3
        a = amem(rand(rng, UInt64, n))
        r = Memory{UInt64}(undef, 2n)
        sqr!(r, 0, a, 0, n)
        @test atoref(r, 0, 2n) == atoref(a, 0, n)^2
    end
end

@testset "divrem! multi-limb" begin
    rng = MersenneTwister(23)

    function checkdiv(a::Memory{UInt64}, n, d::Memory{UInt64}, m)
        aref = atoref(a, 0, n); dref = atoref(d, 0, m)
        q = Memory{UInt64}(undef, n - m + 1)
        r = Memory{UInt64}(undef, m)
        divrem!(q, 0, r, 0, a, 0, n, d, 0, m)
        @test atoref(q, 0, n - m + 1) == aref ÷ dref
        @test atoref(r, 0, m) == aref % dref
    end

    # random sweep over sizes, normalized and unnormalized divisors
    for trial in 1:200
        n = rand(rng, 1:40); m = rand(rng, 1:n)
        a = amem(rand(rng, UInt64, n))
        d = amem(rand(rng, UInt64, m))
        d[m] == 0 && (d[m] = UInt64(1))
        rand(rng) < 0.5 && (d[m] |= UInt64(1) << 63)   # normalized divisor path
        checkdiv(a, n, d, m)
    end

    # adversarial: all-ones numerator, minimal/maximal normalized divisors
    for (n, m) in ((5, 2), (8, 3), (12, 7), (3, 3), (9, 2))
        ones_a = amem(fill(typemax(UInt64), n))
        for dv in (vcat(zeros(UInt64, m - 1), UInt64(1) << 63),       # d = B^m / 2
                   fill(typemax(UInt64), m),                          # d = B^m - 1
                   vcat(fill(typemax(UInt64), m - 1), UInt64(1) << 63))
            checkdiv(ones_a, n, amem(copy(dv)), m)
        end
    end

    # qhat == B-1 special case: numerator top limbs replicate the divisor's
    for trial in 1:50
        m = rand(rng, 2:6); n = m + rand(rng, 1:4)
        dref = atoref(amem(rand(rng, UInt64, m)), 0, m) | (big(1) << (64m - 1))
        # a = d * (B^k - 1) + small ⟹ quotient limbs of typemax
        k = n - m
        aref = dref * ((big(1) << (64k)) - 1) + rand(rng, big(0):dref-1)
        checkdiv(afrombig(aref, n), n, afrombig(dref, m), m)
    end

    # add-back stress: fat low divisor limbs make qhat overshoot as likely as possible
    for trial in 1:300
        m = rand(rng, 3:8); n = m + rand(rng, 1:6)
        dv = fill(typemax(UInt64), m); dv[m] = UInt64(1) << 63
        av = fill(typemax(UInt64), n)
        for i in 1:n
            rand(rng) < 0.3 && (av[i] = rand(rng, UInt64))
        end
        checkdiv(amem(av), n, amem(dv), m)
    end

    # exact multiples and quotient == 0
    for trial in 1:50
        m = rand(rng, 2:8); n = m + rand(rng, 0:6)
        dref = atoref(amem(rand(rng, UInt64, m)), 0, m)
        dref == 0 && continue
        dref |= big(1) << (64 * (m - 1))          # keep m limbs
        qref = rand(rng, big(0):(big(1) << (64 * (n - m))) - 1)
        aref = qref * dref
        aref < big(1) << (64n) || continue
        checkdiv(afrombig(aref, n), n, afrombig(dref, m), m)
        checkdiv(afrombig(dref - 1, m), m, afrombig(dref, m), m)   # a < d ⟹ q = 0
    end
end

@testset "div_blocks! (exact dc)" begin
    rng = MersenneTwister(37)

    # Direct exact-dc div_blocks! check with a forced-low threshold to exercise
    # deep recursion. dref must be normalized (top bit of limb m set); numerator
    # is nn limbs with the top limb possibly nonzero (qh convention as divrem_bc!).
    function checkdc(aref::BigInt, dref::BigInt, nn::Int, m::Int, thr::Int)
        u = afrombig(aref, nn)
        d = afrombig(dref, m)
        v = invert_pi1(d[m], d[m-1])
        qn = nn - m
        q = Memory{UInt64}(undef, qn)
        qh = div_blocks!(q, 0, u, 0, nn, d, 0, m, DcEngine(v, thr), false)
        qref, rref = divrem(aref, dref)
        @test (big(qh) << (64qn)) + atoref(q, 0, qn) == qref
        @test atoref(u, 0, m) == rref
    end

    randnorm(m) = atoref(amem(rand(rng, UInt64, m)), 0, m) | (big(1) << (64m - 1))

    # balanced (qn == m): random sweep, deep recursion via thr = 4
    for trial in 1:100
        m = rand(rng, 4:40)
        dref = randnorm(m)
        aref = rand(rng, big(0):(big(1) << (64 * 2m)) - 1)
        checkdc(aref, dref, 2m, m, 4)
    end

    # qn < m: truncate-and-correct path (needs qn >= thr to take the dc branch)
    for trial in 1:100
        m = rand(rng, 9:40)
        qn = rand(rng, 4:m-1)
        dref = randnorm(m)
        aref = rand(rng, big(0):(big(1) << (64 * (m + qn))) - 1)
        checkdc(aref, dref, m + qn, m, 4)
    end

    # qn > m: outer block loop, all leading-block sizes s = 1..m
    for trial in 1:100
        m = rand(rng, 4:16)
        qn = m + rand(rng, 1:3m)
        dref = randnorm(m)
        aref = rand(rng, big(0):(big(1) << (64 * (m + qn))) - 1)
        checkdc(aref, dref, m + qn, m, 4)
    end

    # qh = 1: numerator's top m limbs >= d
    for trial in 1:50
        m = rand(rng, 4:20)
        qn = rand(rng, 4:2m)
        dref = randnorm(m)
        aref = (dref << (64qn)) + rand(rng, big(0):(big(1) << (64qn)) - 1)
        checkdc(aref, dref, m + qn, m, 4)
    end

    # add-back stress: all-ones divisor tails and numerators drive the
    # block-correction (q -= 1, r += d) loops as hard as possible
    for trial in 1:200
        m = rand(rng, 4:24)
        qn = rand(rng, 4:2m)
        dv = fill(typemax(UInt64), m); dv[m] = UInt64(1) << 63
        rand(rng) < 0.5 && (dv[m] = typemax(UInt64))
        av = fill(typemax(UInt64), m + qn)
        for i in 1:m+qn
            rand(rng) < 0.25 && (av[i] = rand(rng, UInt64))
        end
        checkdc(atoref(amem(av), 0, m + qn), atoref(amem(dv), 0, m), m + qn, m, 4)
    end

    # exact multiples and tiny remainders
    for trial in 1:50
        m = rand(rng, 4:16)
        qn = rand(rng, 4:2m)
        dref = randnorm(m)
        qref = rand(rng, big(0):(big(1) << (64qn)) - 1)
        checkdc(qref * dref + rand(rng, big(0):big(1)), dref, m + qn, m, 4)
    end

    # production threshold: divrem! dispatches to dc above DC_DIV_THRESHOLD;
    # cross-check against BigInt at sizes straddling and well above it
    function checkdiv(a::Memory{UInt64}, n, d::Memory{UInt64}, m)
        aref = atoref(a, 0, n); dref = atoref(d, 0, m)
        q = Memory{UInt64}(undef, n - m + 1)
        r = Memory{UInt64}(undef, m)
        divrem!(q, 0, r, 0, a, 0, n, d, 0, m)
        @test atoref(q, 0, n - m + 1) == aref ÷ dref
        @test atoref(r, 0, m) == aref % dref
    end
    thr = DC_DIV_THRESHOLD
    for (n, m) in ((2thr, thr), (2thr + 1, thr + 1), (4thr, 2thr), (6thr, thr + 3),
                   (3thr, 2thr), (8thr, 3thr))
        a = amem(rand(rng, UInt64, n))
        d = amem(rand(rng, UInt64, m))
        d[m] == 0 && (d[m] = UInt64(1))
        rand(rng) < 0.5 && (d[m] |= UInt64(1) << 63)   # unnormalized divisor path too
        checkdiv(a, n, d, m)
    end
end

using NativeBigInt: mu_reduce_setup, mu_reduce!, powermod_limbs,
    BARRETT_THRESHOLD, BARRETT_EVEN_THRESHOLD

@testset "invertappr!" begin
    B = big(2)^64
    # V = ⌊(β^2n - 1)/A⌋ - β^n; the -1 keeps V inside n limbs when A = β^n/2.
    exactinv(A, n) = fld(B^(2n) - 1, A) - B^n
    rng = MersenneTwister(20260822)
    maxerr = big(0)

    for n in vcat(1:64, 70:7:200, [255, 256, 257])
        lo = B^n >> 1          # minimal normalized
        hi = B^n - 1           # maximal
        cands = BigInt[lo, hi, lo + 1, hi - 1, B^n - B^(n-1), lo | 1]
        append!(cands, [rand(rng, lo:hi) for _ in 1:6])
        for A in cands
            @assert lo <= A <= hi          # contract: A normalized
            d = afrombig(A, n)
            V = exactinv(A, n)
            # thr = 4 is the deepest recursion the clamp allows, thr = n forces
            # the direct-division basecase; both must hold the contract at every
            # size, independent of where INV_NEWTON_THRESHOLD sits.
            for thr in (4, 5, n, INV_NEWTON_THRESHOLD)
                ip = Memory{UInt64}(undef, n)
                sc = Memory{UInt64}(undef, invertappr_scratch_len(n, thr))
                invertappr!(ip, 0, d, 0, n, sc, 0, thr)
                err = V - atoref(ip, 0, n)
                @test 0 <= err <= 1
                @test atoref(d, 0, n) == A     # d is read-only
                maxerr = max(maxerr, err)
            end
            # the basecase divides exactly, so it should be error-free
            ipb = Memory{UInt64}(undef, n)
            invertappr!(ipb, 0, d, 0, n, nothing, 0, n)
            @test atoref(ipb, 0, n) == V
        end
    end
    @test maxerr <= 1

    # n == 1 is exactly invert_limb: no error at all
    for _ in 1:64
        A = rand(rng, (B >> 1):(B - 1))
        ip = Memory{UInt64}(undef, 1)
        invertappr!(ip, 0, afrombig(A, 1), 0, 1)
        @test atoref(ip, 0, 1) == exactinv(A, 1)
    end

    # invertappr_step!: refine a half-width seed instead of rebuilding from
    # scratch. This is the entry point a caller carrying a reciprocal up a
    # recursion uses, so it must hold the same ≤ 1 ulp contract on its own, and
    # above the basecase threshold it must reproduce invertappr! exactly (same
    # code path, just with the recursion supplied by the caller).
    for n in (5, 6, 7, 16, 33, 64, 73, 100, 129, 200, 257)
        A = rand(rng, (B^n >> 1):(B^n - 1))
        d = afrombig(A, n)
        hs = NativeBigInt.invertappr_seed_len(n)
        l = n - hs
        ip = Memory{UInt64}(undef, n)
        invertappr!(ip, l, d, l, hs)          # seed: reciprocal of d's top hs limbs
        # the step's own sizing — invertappr_scratch_len would give the
        # basecase size for n <= thr and be too small
        sc = Memory{UInt64}(undef, NativeBigInt.invertappr_step_scratch_len(n))
        NativeBigInt.invertappr_step!(ip, 0, d, 0, n, sc, 0)
        @test 0 <= exactinv(A, n) - atoref(ip, 0, n) <= 1
        if n > INV_NEWTON_THRESHOLD
            ref = Memory{UInt64}(undef, n)
            invertappr!(ref, 0, d, 0, n)
            @test atoref(ip, 0, n) == atoref(ref, 0, n)
        end
    end

    # non-zero offsets and a caller-supplied scratch match the plain call
    for n in (2, 3, 17, 64, 91)
        A = rand(rng, (B^n >> 1):(B^n - 1))
        d = afrombig(A, n)
        ref = Memory{UInt64}(undef, n)
        invertappr!(ref, 0, d, 0, n)

        doff, ioff, soff = 5, 3, 7
        dd = Memory{UInt64}(undef, n + doff)
        fill!(dd, typemax(UInt64))
        copyto!(dd, doff + 1, d, 1, n)
        ii = Memory{UInt64}(undef, n + ioff)
        fill!(ii, typemax(UInt64))
        sc = Memory{UInt64}(undef, invertappr_scratch_len(n) + soff)
        invertappr!(ii, ioff, dd, doff, n, sc, soff)
        @test atoref(ii, ioff, n) == atoref(ref, 0, n)
        @test atoref(dd, doff, n) == A
    end
end

@testset "mu_divrem!" begin
    rng = MersenneTwister(31337)
    # (n, m) shapes: balanced 2m/m, long quotient, short quotient, ragged
    shapes = Tuple{Int,Int}[]
    for m in (2, 3, 5, 8, 17, 33, 64, 100, 129, 200)
        push!(shapes, (2m, m), (2m + 1, m), (m + 1, m), (3m, m), (m + 2, m))
        m >= 4 && push!(shapes, (5m, m), (2m - 1, m))
    end
    for (n, m) in shapes
        n <= m && continue
        for trial in 1:6
            A = rand(rng, big(1):(big(2)^(64n) - 1))
            # exercise both normalized and unnormalized divisors
            D = trial <= 3 ? rand(rng, (big(2)^(64m - 1)):(big(2)^(64m) - 1)) :
                             rand(rng, (big(2)^(64 * (m - 1))):(big(2)^(64m - 1) - 1))
            D == 0 && continue
            a = afrombig(A, n)
            d = afrombig(D, m)
            q = Memory{UInt64}(undef, n - m + 1)
            r = Memory{UInt64}(undef, m)
            mu_divrem!(q, 0, r, 0, a, 0, n, d, 0, m)
            @test atoref(q, 0, n - m + 1) == fld(A, D)
            @test atoref(r, 0, m) == mod(A, D)
            @test atoref(a, 0, n) == A          # a is read-only
            @test atoref(d, 0, m) == D
        end
    end

    # agrees with divrem! limb for limb, including a caller-supplied scratch
    for (n, m) in ((64, 32), (129, 64), (200, 100), (301, 100))
        A = rand(rng, big(1):(big(2)^(64n) - 1))
        D = rand(rng, (big(2)^(64m - 1)):(big(2)^(64m) - 1))
        a = afrombig(A, n); d = afrombig(D, m)
        q1 = Memory{UInt64}(undef, n - m + 1); r1 = Memory{UInt64}(undef, m)
        q2 = Memory{UInt64}(undef, n - m + 1); r2 = Memory{UInt64}(undef, m)
        divrem!(q1, 0, r1, 0, a, 0, n, d, 0, m)
        sc = Memory{UInt64}(undef, NativeBigInt.mu_divrem_scratch_len(n, m) + 5)
        mu_divrem!(q2, 0, r2, 0, a, 0, n, d, 0, m, sc, 5)
        @test atoref(q1, 0, n - m + 1) == atoref(q2, 0, n - m + 1)
        @test atoref(r1, 0, m) == atoref(r2, 0, m)
    end

    # divrem!'s Barrett dispatch: the predicate, and that routing through it
    # still agrees with BigInt (nothing else in the suite reaches m >= 256)
    @test NativeBigInt.mu_div_worthwhile(4 * 256 + 1, 256)
    @test !NativeBigInt.mu_div_worthwhile(257, 256)      # k=1 at m=256: dc wins
    @test !NativeBigInt.mu_div_worthwhile(2 * 256, 256)  # k=2 at m=256: dc wins
    @test NativeBigInt.mu_div_worthwhile(2 * 384, 384)
    @test NativeBigInt.mu_div_worthwhile(1025, 1024)
    # short quotients below the qn crossover must never dispatch, at any m:
    # mu degrades without limit as qn shrinks (100x at m=1024, qn=2 when this
    # was unguarded), since the estimate cost does not shrink with it
    for m in (256, 384, 1024, 4096, 16384), qn in (1, 2, 16, 64, 256, 512, 640)
        qn < m && @test !NativeBigInt.mu_div_worthwhile(qn, m)
    end
    # ... but a long-enough quotient does dispatch even when shorter than the
    # divisor: the deciding sub-problem is qn, not m
    @test NativeBigInt.mu_div_worthwhile(768, 1024)
    @test NativeBigInt.mu_div_worthwhile(2048, 4096)
    @test !NativeBigInt.mu_div_worthwhile(512, 4096)
    @test NativeBigInt.mu_div_worthwhile(4096, 4096)
    for (n, m) in ((5 * 256, 256), (2 * 1024, 1024))
        A = rand(rng, big(1):(big(2)^(64n) - 1))
        D = rand(rng, (big(2)^(64m - 1)):(big(2)^(64m) - 1))
        a = afrombig(A, n)
        d = afrombig(D, m)
        q = Memory{UInt64}(undef, n - m + 1)
        r = Memory{UInt64}(undef, m)
        divrem!(q, 0, r, 0, a, 0, n, d, 0, m)
        @test atoref(q, 0, n - m + 1) == fld(A, D)
        @test atoref(r, 0, m) == mod(A, D)
    end

    # adversarial divisors: minimal/maximal normalized, and a power of the base
    for m in (4, 16, 64, 130), n in (2m, 3m + 1)
        for D in (big(2)^(64m - 1), big(2)^(64m) - 1, big(2)^(64m) - big(2)^(64m - 1))
            A = rand(rng, big(1):(big(2)^(64n) - 1))
            a = afrombig(A, n); d = afrombig(D, m)
            q = Memory{UInt64}(undef, n - m + 1); r = Memory{UInt64}(undef, m)
            mu_divrem!(q, 0, r, 0, a, 0, n, d, 0, m)
            @test atoref(q, 0, n - m + 1) == fld(A, D)
            @test atoref(r, 0, m) == mod(A, D)
        end
    end
end

@testset "mu_reduce!" begin
    rng = MersenneTwister(0xba44e77)

    # mu_reduce! requires T < m·β^k (the mu block's U < d·β^s at s = k), which
    # is narrower than the general T < β^2k the old hand-rolled Barrett took: a
    # caller with arbitrary T must go through divrem!. Every T below respects
    # it, and the helper asserts it rather than trusting the construction.
    # k >= 2 likewise: the reduction is only reachable above BARRETT_THRESHOLD.
    function checkmu(Tref::BigInt, mref::BigInt, k::Int)
        @assert k >= 2 && Tref < mref * (big(1) << 64k)
        mbuf = afrombig(mref, k)
        st = mu_reduce_setup(mbuf, 0, k)
        @test atoref(st.mp, 0, k) == mref << leading_zeros(mbuf[k])  # normalized
        r = Memory{UInt64}(undef, k)
        mu_reduce!(r, 0, afrombig(Tref, 2k), 0, k, st)
        @test atoref(r, 0, k) == mod(Tref, mref)
        @test atoref(mbuf, 0, k) == mref                             # m read-only
    end

    # the real usage shape: T = x·y with x, y < m
    for trial in 1:200
        k = rand(rng, 2:40)
        mref = rand(rng, big(1) << (64k - 64):(big(1) << 64k) - 1)
        mref <= 1 && (mref = big(2))
        checkmu(rand(rng, big(0):mref-1) * rand(rng, big(0):mref-1), mref, k)
    end

    # maximal quotient (q = β^k - 1 is the largest T < m·β^k allows) with the
    # remainder pinned at each end, to drive the 0/1/2-correction arms
    for trial in 1:100
        k = rand(rng, 2:30)
        mref = rand(rng, big(1) << (64k - 64):(big(1) << 64k) - 1)
        mref <= 1 && (mref = big(3))
        q = (big(1) << 64k) - 1
        checkmu(q * mref, mref, k)                       # r = 0
        checkmu(q * mref + rand(rng, big(0):mref-1), mref, k)
        checkmu(q * mref + mref - 1, mref, k)            # r = m - 1
        checkmu((q - 1) * mref + mref - 1, mref, k)
    end

    # edges: m with minimal top limb (largest normalizing shift), m = β^(k-1),
    # m = β^k - 1 (shift 0); T at the top of the contract, T < m, T = 0
    for k in (2, 3, 7, 20)
        small_top = (big(1) << (64k - 64)) | rand(rng, big(0):(big(1) << (64k - 64)) - 1)
        for mref in (small_top, big(1) << (64k - 64), (big(1) << 64k) - 1)
            mref <= 1 && continue
            checkmu(mref * (big(1) << 64k) - 1, mref, k) # largest legal T
            checkmu(big(0), mref, k)
            checkmu(mref - 1, mref, k)
            checkmu(mref + 1, mref, k)
            checkmu((mref - 1)^2, mref, k)
        end
    end
end

@testset "powermod_limbs Barrett path" begin
    rng = MersenneTwister(0xb42)
    # force the Barrett branch at small k (cheap) and cross-check vs BigInt,
    # odd and even moduli, exponents crossing the window-size breakpoints
    for trial in 1:60
        k = rand(rng, 1:25)
        mref = rand(rng, big(2):(big(1) << 64k) - 1)
        mref |= big(1) << (64k - 64)                    # keep k limbs
        trial % 2 == 0 && iseven(mref) && (mref += 1)   # both parities
        bref = rand(rng, big(1):mref-1)
        e = rand(rng, big(1):big(2)^rand(rng, (5, 30, 100)))
        lb = max(cld(ndigits(bref, base=2), 64), 1)
        r = powermod_limbs(afrombig(bref, lb), lb, e, afrombig(mref, k), k, true)
        @test atoref(r, 0, k) == powermod(bref, e, mref)
    end
    # production dispatch at the per-parity thresholds, small exponent
    for (k, low) in ((BARRETT_THRESHOLD, 1), (BARRETT_EVEN_THRESHOLD, 0))
        mref = rand(rng, big(1) << (64k - 1):(big(1) << 64k) - 1)
        mref = (mref & ~big(1)) | low
        bref = rand(rng, big(1):mref-1)
        r = powermod_limbs(afrombig(bref, k), k, big(65537), afrombig(mref, k), k)
        @test atoref(r, 0, k) == powermod(bref, big(65537), mref)
    end
end

using NativeBigInt: HgcdMatrix, hgcd_matrix_cap, hgcd!, gcd!, gcdext!, normlen

@testset "hgcd" begin
    rng = MersenneTwister(0x59cd)

    # hgcd! contract, BigInt-verified: (A; B) == M * (a'; b') exactly,
    # det(M) == +1, both outputs > s = n÷2 + 1 limbs, and nn == 0 leaves M
    # the identity. Tiny thresholds force deep recursion on small inputs.
    mval(M) = (atoref(M.m00, 0, M.n), atoref(M.m01, 0, M.n),
               atoref(M.m10, 0, M.n), atoref(M.m11, 0, M.n))
    for trial in 1:600
        n = rand(rng, 3:40)
        thr = rand(rng, (4, 8, 1000))
        a0 = rand(rng, big(1):big(2)^(64n) - 1)
        b0 = rand(rng, big(1):big(2)^(64n) - 1)
        trial % 5 == 0 && (b0 = rand(rng, big(1):big(2)^(64 * max(n ÷ 2, 1)) - 1))
        trial % 7 == 0 && (b0 = max(a0 - rand(rng, big(1):big(2)^32), big(1)))
        n = max(cld(max(ndigits(a0, base=2), ndigits(b0, base=2)), 64), 1)
        a = afrombig(a0, n + 1)
        b = afrombig(b0, n + 1)
        M = HgcdMatrix(hgcd_matrix_cap(n))
        ra, rb, rn = hgcd!(a, b, n, M, thr)
        m00, m01, m10, m11 = mval(M)
        if rn == 0
            @test (m00, m01, m10, m11) == (1, 0, 0, 1)
        else
            s = n ÷ 2 + 1
            an = normlen(ra, 0, rn)
            bn = normlen(rb, 0, rn)
            av = atoref(ra, 0, an)
            bv = atoref(rb, 0, bn)
            @test m00 * av + m01 * bv == a0
            @test m10 * av + m11 * bv == b0
            @test m00 * m11 - m01 * m10 == 1
            @test an > s && bn > s
        end
    end

    # DC driver differential vs BigInt with tiny thresholds (deep recursion),
    # including planted common factors, Fibonacci pairs (all-quotient-1
    # chains), and quotient spikes (subdiv + q-1 guard).
    for trial in 1:300
        la, lb = rand(rng, 1:50), rand(rng, 1:50)
        a0 = rand(rng, big(1):big(2)^(64la) - 1)
        b0 = rand(rng, big(1):big(2)^(64lb) - 1)
        if isodd(trial)
            g = rand(rng, big(1):big(2)^(64 * rand(rng, 1:8)))
            a0 *= g; b0 *= g
        end
        if trial % 11 == 0
            x, y = big(1), big(1)
            while ndigits(y, base=2) < 64la
                x, y = y, x + y
            end
            a0, b0 = y, x
        end
        trial % 13 == 0 &&
            (b0 = a0 * rand(rng, big(2)^200:big(2)^220) + rand(rng, big(1):a0))
        cap = max(cld(ndigits(a0, base=2), 64), cld(ndigits(b0, base=2), 64)) + 1
        lu = cld(ndigits(a0, base=2), 64)
        lv = cld(ndigits(b0, base=2), 64)
        g1, lg = gcd!(afrombig(a0, cap), lu, afrombig(b0, cap), lv;
                      dc_thr=10, hgcd_thr=6)
        @test atoref(g1, 0, lg) == gcd(a0, b0)
        g2, lg, tm, lt, tpos = gcdext!(afrombig(a0, cap + 3), lu,
                                       afrombig(b0, cap + 3), lv;
                                       dc_thr=10, hgcd_thr=6)
        gv = atoref(g2, 0, lg)
        tv = atoref(tm, 0, lt) * (tpos ? 1 : -1)
        @test gv == gcd(a0, b0)
        @test mod(gv - tv * b0, a0) == 0        # Bézout: s*a + t*b == g
        @test abs(tv) <= max(a0, b0) ÷ gv || gv == b0
    end

    # production thresholds: sizes straddling GCDEXT_DC/GCD_DC and well above
    for n in (280, 320, 500, 800)
        a0 = rand(rng, big(1):big(2)^(64n) - 1)
        b0 = rand(rng, big(1):big(2)^(64n) - 1)
        g = rand(rng, big(1):big(2)^320)
        a0 *= g; b0 *= g
        cap = max(cld(ndigits(a0, base=2), 64), cld(ndigits(b0, base=2), 64)) + 1
        lu = cld(ndigits(a0, base=2), 64)
        lv = cld(ndigits(b0, base=2), 64)
        g1, lg = gcd!(afrombig(a0, cap), lu, afrombig(b0, cap), lv)
        @test atoref(g1, 0, lg) == gcd(a0, b0)
        g2, lg, tm, lt, tpos = gcdext!(afrombig(a0, cap + 3), lu,
                                       afrombig(b0, cap + 3), lv)
        gv = atoref(g2, 0, lg)
        tv = atoref(tm, 0, lt) * (tpos ? 1 : -1)
        @test gv == gcd(a0, b0)
        @test mod(gv - tv * b0, a0) == 0
    end
end

@testset "div_blocks! (approx)/divappr_bc! approximate quotient" begin
    using NativeBigInt: divappr_bc!, invert_pi1, lshift!,
        magnitude_bits, mu_block_scratch_len, invertappr!,
        invertappr_scratch_len
    rng = MersenneTwister(0xd1ab)
    hib = UInt64(1) << 63
    β = big(1) << 64

    # Driver for the divappr engines over arbitrary shapes: entry divisor
    # truncation, normalization, then the dc/bc dispatch. Production has no
    # such wrapper — sqrt.jl calls the engines directly from its own normalized
    # buffer (sqrt.jl:238-248) — so the test supplies one rather than leaving
    # the engines covered only indirectly through isqrt.
    function divappr_ref!(q, qo, a, ao, n, d, do_, m; mu::Bool=false)
        qn = n - m + 1
        if m > qn + 2
            drop = m - (qn + 2)
            return divappr_ref!(q, qo, a, ao + drop, n - drop, d, do_ + drop,
                                m - drop; mu = mu)
        end
        scratch = Memory{UInt64}(undef, n + 1 + 3m + m +
                                        max(invertappr_scratch_len(m),
                                            mu_block_scratch_len(m)))
        if m <= 2 || magnitude_bits(a, ao, n) - magnitude_bits(d, do_, m) <= 2
            return divrem!(q, qo, scratch, 0, a, ao, n, d, do_, m, scratch, m)
        end
        l = leading_zeros(d[do_+m])
        nn = n + 1
        if l == 0
            copyto!(scratch, 1, a, ao + 1, n)
            scratch[nn] = zero(UInt64)
            dv, dvo = d, do_
        else
            scratch[nn] = lshift!(scratch, 0, a, ao, n, l)
            lshift!(scratch, nn, d, do_, m, l)
            dv, dvo = scratch, nn
        end
        if mu
            # production always supplies the reciprocal (sqrt's ladder); build
            # one here so the engine can be exercised over arbitrary shapes
            invertappr!(scratch, nn + m, dv, dvo, m, scratch, nn + 2m)
            div_blocks!(q, qo, scratch, 0, nn, dv, dvo, m,
                        MuEngine(scratch, nn + m, m), true, scratch, nn + 2m)
            return nothing
        end
        v = invert_pi1(dv[dvo+m], dv[dvo+m-1])
        if m >= DC_DIV_THRESHOLD && nn - m >= DC_DIV_THRESHOLD
            div_blocks!(q, qo, scratch, 0, nn, dv, dvo, m,
                        DcEngine(v), true, scratch, nn + m)
        else
            divappr_bc!(q, qo, scratch, 0, nn, dv, dvo, m, v)
        end
        return nothing
    end

    # q̂ - floor(a/d): the contract is one-sided over-approximation by at most
    # 32 ulps, with a unmodified and no remainder computed.
    function apprerr(aref::BigInt, n, dref::BigInt, m; mu::Bool=false)
        a = afrombig(aref, n)
        acopy = copy(a)
        d = afrombig(dref, m)
        q = Memory{UInt64}(undef, n - m + 1)
        divappr_ref!(q, 0, a, 0, n, d, 0, m; mu = mu)
        @test a == acopy
        return atoref(q, 0, n - m + 1) - aref ÷ dref
    end

    maxerr = big(0)
    maxmuerr = big(0)
    function checkshape(n, m, dref, aref)
        err = apprerr(aref, n, dref, m)
        @test 0 <= err <= 32
        maxerr = max(maxerr, err)
        # the mu engine over the same shapes: same one-sided contract, tighter
        # bound (its only inexact block undershoots by <= 5, lifted by +5)
        if m >= 3 && n > m
            muerr = apprerr(aref, n, dref, m; mu = true)
            @test 0 <= muerr <= 6
            maxmuerr = max(maxmuerr, muerr)
        end
    end

    # small random sweep: m = 1, 2, small-quotient path, unnormalized divisors
    for trial in 1:150
        n = rand(rng, 1:30); m = rand(rng, 1:n)
        dref = rand(rng, big(1):β^m - 1)
        dref >= β^(m - 1) || (dref += β^(m - 1))
        checkshape(n, m, dref, rand(rng, big(0):β^n - 1))
    end

    # dc shapes: threshold-straddling, balanced 2m/m, sqrt shape (qn ≈ m + 2),
    # short quotient over long divisor (entry truncation), deep recursion
    shapes = [(199, 100), (200, 101), (240, 120), (300, 150), (700, 350),
              (260, 129), (262, 130),               # sqrt shape qn = m + 2
              (349, 300), (420, 400), (150, 130),   # qn ≪ m entry truncation
              (511, 128), (1000, 128)]              # multi-block peeling
    for (n, m) in shapes, trial in 1:6
        dref = rand(rng, big(0):β^m - 1) | (big(1) << (64m - 64))
        rand(rng) < 0.5 && (dref |= big(hib) << (64 * (m - 1)))
        aref = rand(rng, big(0):β^n - 1)
        checkshape(n, m, dref, aref)
        # adversarial low divisor mass + boundary numerators
        dadv = (big(hib) << (64 * (m - 1))) | (β^(m - 1) - 1)
        k = rand(rng, big(0):β^(n - m) - 1)
        for w in (big(0), big(1), dadv - 1)
            k * dadv + w < β^n && checkshape(n, m, dadv, k * dadv + w)
        end
    end

    # exact multiples and all-ones stress at dc size
    for trial in 1:10
        m = 128; n = 256
        dref = rand(rng, big(0):β^m - 1) | (big(hib) << (64 * (m - 1)))
        qref = rand(rng, big(0):β^(n - m) - 1)
        qref * dref < β^n && checkshape(n, m, dref, qref * dref)
        checkshape(n, m, dref, β^n - 1)
    end
    @test maxerr <= 32
    @test maxmuerr <= 6
end
