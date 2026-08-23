# Differential property tests vs Base.BigInt: random + adversarial operands,
# all sign combinations, bit-for-bit agreement.

# Random n-limb magnitude with adversarial patterns: uniform limbs, all-ones
# limbs (carry chains), 2^k ± 1, single high bit.
function diff_randbig(rng, n::Int)
    n == 0 && return big(0)
    style = rand(rng, 1:4)
    mag = if style == 1
        rand(rng, big(2)^(64 * (n - 1)):big(2)^(64n)-1)
    elseif style == 2
        big(2)^(64n) - 1
    elseif style == 3
        big(2)^rand(rng, (64 * (n - 1)):(64n - 1)) + rand(rng, -1:1)
    else
        big(2)^(64n - 1)
    end
    return rand(rng, Bool) ? -mag : mag
end

@testset "differential isqrt" begin
    rng = MersenneTwister(0x1509)
    for x in 0:20
        @test BigInt(isqrt(NBig(x))) == isqrt(big(x))
    end
    @test_throws DomainError isqrt(NBig(-1))
    for trial in 1:300
        n = rand(rng, 1:12)
        trial % 20 == 0 && (n = rand(rng, 30:100))
        # mid band: divappr top level at recursion depths 2-4
        trial % 15 == 0 && (n = rand(rng, 80:340))
        # deep divappr certificate path (dc-sized top-level division);
        # a² lands on the always-fallback boundary
        trial % 50 == 0 && (n = rand(rng, 400:560))
        a = abs(diff_randbig(rng, n))
        for x in (a, a^2, a^2 - 1, a^2 + 1)
            s = isqrt(NBig(x))
            @test BigInt(s) == isqrt(x)
        end
    end
end

@testset "isqrt Barrett top level" begin
    # Two paths past the sweep above, which tops out at 560 limbs. hh ≈ n/4, so
    # n ≈ 4100 puts the top level on Barrett with the reciprocal ladder at its
    # from-scratch base (sqrt_build_inv!'s !seeded arm). The *seeded* arm — the
    # widen-then-double step — needs two consecutive Barrett levels, i.e.
    # hh ≥ MU_SQRT_THRESHOLD two levels running, which starts at n ≈ 8192.
    rng = MersenneTwister(0x5137)
    for n in (4100, 8200)
        a = abs(diff_randbig(rng, n))
        for x in (a, a^2 - 1, a^2)
            @test BigInt(isqrt(NBig(x))) == isqrt(x)
        end
    end
end

@testset "differential gcd" begin
    rng = MersenneTwister(0x9cd)
    @test iszero(gcd(NBig(0), NBig(0)))
    @test BigInt(gcd(NBig(0), NBig(-6))) == 6
    @test BigInt(gcd(NBig(-4), NBig(0))) == 4
    for trial in 1:300
        la, lb = rand(rng, 1:12), rand(rng, 1:12)
        trial % 20 == 0 && (la = rand(rng, 30:70); lb = rand(rng, 1:70))
        a, b = diff_randbig(rng, la), diff_randbig(rng, lb)
        # plant a large common factor half the time
        if isodd(trial)
            g = abs(diff_randbig(rng, rand(rng, 1:6)))
            a, b = a * g, b * g
        end
        @test BigInt(gcd(NBig(a), NBig(b))) == gcd(a, b)
        @test BigInt(gcd(NBig(a), NBig(a))) == abs(a)
        @test BigInt(gcd(NBig(a * b), NBig(b))) == gcd(a * b, b)
    end

    @test which(gcdx, Tuple{NBig, NBig}).module === NativeBigInt
    @test (big(gcdx(NBig(0), NBig(0))[1]), ) == (gcdx(big(0), big(0))[1], )
    for trial in 1:200
        a = diff_randbig(rng, rand(rng, 0:8))
        b = diff_randbig(rng, rand(rng, 0:8))
        g, x, y = gcdx(NBig(a), NBig(b))
        @test (big(g), big(x), big(y)) == gcdx(a, b)
    end

    # subquadratic (HGCD) sizes: straddle GCDEXT_DC/GCD_DC and go well above
    for (trial, n) in enumerate((260, 310, 450, 700))
        a = diff_randbig(rng, n)
        b = diff_randbig(rng, rand(rng, n ÷ 2:n))
        if isodd(trial)
            g = abs(diff_randbig(rng, 6))
            a, b = a * g, b * g
        end
        @test BigInt(gcd(NBig(a), NBig(b))) == gcd(a, b)
        g, x, y = gcdx(NBig(a), NBig(b))
        @test (big(g), big(x), big(y)) == gcdx(a, b)
    end
end

@testset "differential powermod" begin
    rng = MersenneTwister(0x90d)
    @test_throws DivideError powermod(NBig(2), NBig(5), NBig(0))
    @test BigInt(powermod(NBig(2), NBig(-1), NBig(7))) == 4
    @test_throws DomainError powermod(NBig(2), NBig(-1), NBig(8))  # not coprime
    @test BigInt(powermod(NBig(0), NBig(0), NBig(7))) == 1
    for trial in 1:150
        lm = rand(rng, 1:8)
        trial % 15 == 0 && (lm = rand(rng, 20:40))
        m = diff_randbig(rng, lm)
        iszero(m) && (m = big(3))
        a = diff_randbig(rng, rand(rng, 0:lm+2))
        n = abs(diff_randbig(rng, rand(rng, 1:3)))
        @test BigInt(powermod(NBig(a), NBig(n), NBig(m))) == powermod(a, n, m)
        @test BigInt(powermod(NBig(a), 17, NBig(m))) == powermod(a, 17, m)
        e = rand(rng, 0:40)
        @test BigInt(powermod(NBig(a), NBig(e), NBig(m))) == powermod(a, big(e), m)
        m2 = big(2)^rand(rng, 1:200) # even / power-of-two modulus path
        @test BigInt(powermod(NBig(a), NBig(n), NBig(m2))) == powermod(a, n, m2)
        # non-NBig base with NBig modulus routes to the Montgomery method
        @test BigInt(powermod(3, NBig(n), NBig(m))) == powermod(3, n, m)
    end
    # moduli spanning the per-parity Barrett thresholds so both reduction
    # regimes run end to end; small exponents keep it cheap
    for (parity, thr) in ((1, NativeBigInt.BARRETT_THRESHOLD),
                          (0, NativeBigInt.BARRETT_EVEN_THRESHOLD))
        for lm in (thr - 1, thr, thr + thr ÷ 4)
            m = abs(diff_randbig(rng, lm))
            m <= 2 && (m = big(3))
            m % 2 == parity || (m += 1)
            a = abs(diff_randbig(rng, rand(rng, 1:lm)))
            n = abs(diff_randbig(rng, 1))
            @test BigInt(powermod(NBig(a), NBig(n), NBig(m))) == powermod(a, n, m)
        end
    end
end

@testset "differential widen/top_set_bit/checked/Bool-rem" begin
    rng = MersenneTwister(0xb17)
    @test widen(NBig) === NBig
    @test Base.top_set_bit(NBig(0)) == 0
    for trial in 1:200
        x = diff_randbig(rng, rand(rng, 0:8))
        y = diff_randbig(rng, rand(rng, 0:8))
        @test Base.top_set_bit(NBig(abs(x))) == Base.top_set_bit(abs(x))
        @test (NBig(x) % Bool) == (x % Bool)
        @test Base.Checked.add_with_overflow(NBig(x), NBig(y)) == (NBig(x + y), false)
        @test Base.Checked.sub_with_overflow(NBig(x), NBig(y)) == (NBig(x - y), false)
        @test Base.Checked.mul_with_overflow(NBig(x), NBig(y)) == (NBig(x * y), false)
    end
end

@testset "differential add/sub/mul/divrem" begin
    rng = MersenneTwister(0xd1ff)
    for trial in 1:500
        la = rand(rng, 0:12)
        lb = rand(rng, 0:12)
        # occasionally cross the Karatsuba threshold
        if trial % 25 == 0
            la, lb = rand(rng, 30:100), rand(rng, 1:100)
        end
        # and the divide-and-conquer division threshold (divisor and quotient
        # both need to clear DC_DIV_THRESHOLD, so keep lb and la - lb large)
        if trial % 50 == 0
            lb = rand(rng, 60:400)
            la = lb + rand(rng, 60:1600)
        end
        a, b = diff_randbig(rng, la), diff_randbig(rng, lb)
        na, nb = NBig(a), NBig(b)
        @test BigInt(na + nb) == a + b
        @test BigInt(na - nb) == a - b
        @test BigInt(na * nb) == a * b
        @test BigInt(na * na) == a * a   # squaring path (a.limbs === b.limbs)
        @test BigInt(na^2) == a^2
        if la <= 4
            e = rand(rng, 0:12)
            @test BigInt(na^e) == a^e
        end
        k = rand(rng, 0:200)
        @test BigInt(na << k) == a << k
        @test BigInt(na >> k) == a >> k
        @test BigInt(na & nb) == a & b
        @test BigInt(na | nb) == a | b
        @test BigInt(xor(na, nb)) == xor(a, b)
        @test BigInt(~na) == ~a
        iszero(a) || @test trailing_zeros(na) == trailing_zeros(a)
        a < 0 || @test count_ones(na) == count_ones(a)
        @test string(na) == string(a)
        @test string(na, base = 16) == string(a, base = 16)
        @test string(na, base = 2, pad = 100) == string(a, base = 2, pad = 100)
        @test string(na, base = 8) == string(a, base = 8)
        @test string(na, base = 4) == string(a, base = 4)
        @test string(na, base = 32) == string(a, base = 32)
        @test parse(NBig, string(a)) == na
        @test hash(na) == hash(a)
        @test Float64(na) == Float64(a)
        if !iszero(b)
            q, r = divrem(na, nb)
            @test BigInt(q) == div(a, b)
            @test BigInt(r) == rem(a, b)
            @test BigInt(div(na, nb)) == div(a, b)
            @test BigInt(rem(na, nb)) == rem(a, b)
            @test BigInt(mod(na, nb)) == mod(a, b)
            @test BigInt(fld(na, nb)) == fld(a, b)
            @test BigInt(cld(na, nb)) == cld(a, b)
        end
    end
end
