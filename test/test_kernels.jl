# Kernel tests
using NativeBigInt: Limb, DLimb, add_n!, sub_n!, add!, sub!, add_into!, cmp_limbs, mul_1!, addmul_1!, submul_1!, mul_basecase!, lshift!, rshift!, divrem_1!, invert_limb, div_2by1, invert_pi1, div_3by2, divrem_bc!
using Random: MersenneTwister

mem(v::Vector{UInt64}) = (m = Memory{UInt64}(undef, length(v)); copyto!(m, v); m)
# reference: limbs -> BigInt
toref(m, off, n) = (x = big(0); for i in n:-1:1; x = (x << 64) | m[off+i]; end; x)

@testset "add/sub kernels" begin
    rng = MersenneTwister(42)
    for n in (1, 2, 3, 7, 30), trial in 1:50
        a = mem(rand(rng, UInt64, n)); b = mem(rand(rng, UInt64, n))
        r = Memory{UInt64}(undef, n)
        c = add_n!(r, 0, a, 0, b, 0, n)
        @test toref(r, 0, n) + (big(c) << (64n)) == toref(a, 0, n) + toref(b, 0, n)
        brw = sub_n!(r, 0, a, 0, b, 0, n)
        @test toref(r, 0, n) - (big(brw) << (64n)) == toref(a, 0, n) - toref(b, 0, n)
    end
    # carry chain: all-ones + 1
    n = 5
    a = mem(fill(typemax(UInt64), n)); b = mem([one(UInt64); zeros(UInt64, n-1)])
    r = Memory{UInt64}(undef, n)
    @test add_n!(r, 0, a, 0, b, 0, n) == 1 && all(iszero, r)
    # mixed length add!/sub!
    la, lb = 6, 2
    a = mem(fill(typemax(UInt64), la)); b = mem(rand(rng, UInt64, lb))
    r = Memory{UInt64}(undef, la)
    c = add!(r, 0, a, 0, la, b, 0, lb)
    @test toref(r, 0, la) + (big(c) << (64la)) == toref(a, 0, la) + toref(b, 0, lb)
    brw = sub!(r, 0, a, 0, la, b, 0, lb)
    @test toref(r, 0, la) == toref(a, 0, la) - toref(b, 0, lb) && brw == 0
    # add_into!
    r2 = mem([typemax(UInt64), typemax(UInt64), UInt64(0)])
    add_into!(r2, 0, 3, mem([UInt64(1)]), 0, 1)
    @test toref(r2, 0, 3) == big(1) << 128
    # aliasing: r === a
    a = mem(rand(rng, UInt64, 4)); aref = toref(a, 0, 4); b = mem(rand(rng, UInt64, 4))
    c = add_n!(a, 0, a, 0, b, 0, 4)
    @test toref(a, 0, 4) + (big(c) << 256) == aref + toref(b, 0, 4)
    # cmp_limbs
    @test cmp_limbs(mem([UInt64(1), UInt64(2)]), 0, 2, mem([UInt64(5)]), 0, 1) == 1
    @test cmp_limbs(mem([UInt64(3)]), 0, 1, mem([UInt64(3)]), 0, 1) == 0
    @test cmp_limbs(mem([UInt64(9), UInt64(1)]), 0, 2, mem([UInt64(0), UInt64(2)]), 0, 2) == -1
end

@testset "add_n! SIMD cold path" begin
    # deterministic cases hitting the s == typemax guard inside 8-limb SIMD blocks
    for n in (8, 16, 24, 30)
        # full ripple: (B^n - 1) + 1 chains a carry through every lane
        a = mem(fill(typemax(UInt64), n)); b = mem([one(UInt64); zeros(UInt64, n - 1)])
        r = Memory{UInt64}(undef, n)
        @test add_n!(r, 0, a, 0, b, 0, n) == 1 && all(iszero, r)
        # typemax-sum lane with no incoming carry: cold path fires, result unchanged
        a2 = mem(fill(UInt64(1), n))
        b2v = fill(UInt64(2), n); b2v[3] = typemax(UInt64) - 1; b2v[min(10, n)] = typemax(UInt64) - 1
        b2 = mem(b2v)
        r2 = Memory{UInt64}(undef, n)
        c2 = add_n!(r2, 0, a2, 0, b2, 0, n)
        @test toref(r2, 0, n) + (big(c2) << (64n)) == toref(a2, 0, n) + toref(b2, 0, n)
        # carry generated in lane 1 chains through typemax lanes mid-block
        av = fill(typemax(UInt64), n)
        bv = zeros(UInt64, n); bv[1] = 1; bv[n] = 5
        a3 = mem(av); b3 = mem(bv)
        r3 = Memory{UInt64}(undef, n)
        c3 = add_n!(r3, 0, a3, 0, b3, 0, n)
        @test toref(r3, 0, n) + (big(c3) << (64n)) == toref(a3, 0, n) + toref(b3, 0, n)
        # carry entering a block whose lane 0 sums to typemax (block-boundary chain)
        a4v = fill(UInt64(7), n); a4v[8] = typemax(UInt64)
        b4v = fill(UInt64(9), n); b4v[8] = 1
        if n > 8
            a4v[9] = typemax(UInt64); b4v[9] = 0
        end
        a4 = mem(a4v); b4 = mem(b4v)
        r4 = Memory{UInt64}(undef, n)
        c4 = add_n!(r4, 0, a4, 0, b4, 0, n)
        @test toref(r4, 0, n) + (big(c4) << (64n)) == toref(a4, 0, n) + toref(b4, 0, n)
    end
end

@testset "mul kernels" begin
    rng = MersenneTwister(1)
    for n in (1, 2, 5, 20), trial in 1:50
        a = mem(rand(rng, UInt64, n)); b = rand(rng, UInt64)
        r = Memory{UInt64}(undef, n)
        c = mul_1!(r, 0, a, 0, n, b)
        @test toref(r, 0, n) + (big(c) << (64n)) == toref(a, 0, n) * b
        r0 = toref(r, 0, n)
        c2 = addmul_1!(r, 0, a, 0, n, b)
        @test toref(r, 0, n) + (big(c2) << (64n)) == r0 + toref(a, 0, n) * b
        r1 = toref(r, 0, n)
        c3 = submul_1!(r, 0, a, 0, n, b)
        @test toref(r, 0, n) - (big(c3) << (64n)) == r1 - toref(a, 0, n) * b
    end
    for (m, n) in ((1,1), (3,2), (7,7), (13,4)), trial in 1:30
        a = mem(rand(rng, UInt64, m)); b = mem(rand(rng, UInt64, n))
        r = Memory{UInt64}(undef, m + n)
        mul_basecase!(r, 0, a, 0, m, b, 0, n)
        @test toref(r, 0, m + n) == toref(a, 0, m) * toref(b, 0, n)
    end
    # max-value stress: (B^m - 1)(B^n - 1)
    a = mem(fill(typemax(UInt64), 4)); b = mem(fill(typemax(UInt64), 3))
    r = Memory{UInt64}(undef, 7)
    mul_basecase!(r, 0, a, 0, 4, b, 0, 3)
    @test toref(r, 0, 7) == toref(a, 0, 4) * toref(b, 0, 3)
end

@testset "sqr_basecase! / sqr!" begin
    rng = MersenneTwister(0x59)
    sqr_basecase! = NativeBigInt.sqr_basecase!
    sqr! = NativeBigInt.sqr!
    # every size 1..45: hits n==1/n==2 paths, paired/unpaired last row, and
    # (via sqr!) both sides of SQR_KARATSUBA_THRESHOLD
    for n in 1:45, trial in 1:10
        a = mem(rand(rng, UInt64, n))
        r = Memory{UInt64}(undef, 2n)
        sqr_basecase!(r, 0, a, 0, n)
        @test toref(r, 0, 2n) == toref(a, 0, n)^2
        fill!(r, 0xdead)
        sqr!(r, 0, a, 0, n)
        @test toref(r, 0, 2n) == toref(a, 0, n)^2
    end
    # adversarial: all-ones (max carry chains), 2^k patterns, offsets
    for n in (1, 2, 3, 7, 8, 16, 33)
        a = mem(fill(typemax(UInt64), n))
        r = Memory{UInt64}(undef, 2n)
        sqr_basecase!(r, 0, a, 0, n)
        @test toref(r, 0, 2n) == toref(a, 0, n)^2
    end
    for n in (60, 100, 157)  # Karatsuba recursion, odd/even splits
        a = mem(rand(rng, UInt64, n))
        r = Memory{UInt64}(undef, 2n)
        sqr!(r, 0, a, 0, n)
        @test toref(r, 0, 2n) == toref(a, 0, n)^2
    end
    # nonzero offsets
    a = mem(rand(rng, UInt64, 9))
    r = Memory{UInt64}(undef, 16)
    sqr_basecase!(r, 2, a, 3, 6)
    @test toref(r, 2, 12) == toref(a, 3, 6)^2
end

@testset "addmul_2! direct" begin
    rng = MersenneTwister(99)
    addmul_2! = NativeBigInt.addmul_2!
    check2(m, av, b0, b1) = begin
        a = mem(av)
        rv = rand(rng, UInt64, m + 2)
        r = mem(copy(rv))
        addmul_2!(r, 0, a, 0, m, b0, b1)
        want = toref(mem(rv), 0, m) + toref(a, 0, m) * (big(b0) + (big(b1) << 64))
        @test toref(r, 0, m + 2) == want
    end
    # widths covering scalar-only, one vector block, blocks + tail
    for m in (1, 2, 7, 8, 9, 16, 23, 24, 31), trial in 1:20
        check2(m, rand(rng, UInt64, m), rand(rng, UInt64), rand(rng, UInt64))
    end
    # cold path: all-ones a with all-ones b limbs maximizes lane sums/carries
    for m in (8, 16, 24, 31)
        check2(m, fill(typemax(UInt64), m), typemax(UInt64), typemax(UInt64))
        # sparse typemax lanes mid-block
        av = fill(UInt64(1), m); av[3] = typemax(UInt64); av[min(10, m)] = typemax(UInt64) - 1
        check2(m, av, typemax(UInt64), UInt64(2))
    end
end

@testset "mul_2! direct" begin
    rng = MersenneTwister(101)
    mul_2! = NativeBigInt.mul_2!
    checkm2(m, av, b0, b1) = begin
        a = mem(av)
        r = mem(rand(rng, UInt64, m + 2))  # garbage: mul_2! must not read r
        mul_2!(r, 0, a, 0, m, b0, b1)
        @test toref(r, 0, m + 2) == toref(a, 0, m) * (big(b0) + (big(b1) << 64))
    end
    for m in (1, 2, 7, 8, 9, 16, 23, 24, 31), trial in 1:20
        checkm2(m, rand(rng, UInt64, m), rand(rng, UInt64), rand(rng, UInt64))
    end
    # cold path / carry saturation
    for m in (8, 16, 24, 31)
        checkm2(m, fill(typemax(UInt64), m), typemax(UInt64), typemax(UInt64))
        av = fill(UInt64(1), m); av[3] = typemax(UInt64); av[min(10, m)] = typemax(UInt64) - 1
        checkm2(m, av, typemax(UInt64), UInt64(2))
    end
end

@testset "addmul_2! carry saturation" begin
    # typemax-dense operands drive the addmul_2! column carry to exactly 2^64
    # (hi0 = 2^64-2 plus two overflow bits), which a limb-sized carry drops
    rng = MersenneTwister(1234)
    for trial in 1:300
        m = rand(rng, 8:23); n = rand(rng, 2:40)
        av = rand(rng, UInt64, m); bv = rand(rng, UInt64, n)
        for i in 1:m; rand(rng) < 0.5 && (av[i] = typemax(UInt64) - rand(rng, 0:1)); end
        for i in 1:n; rand(rng) < 0.5 && (bv[i] = typemax(UInt64) - rand(rng, 0:1)); end
        a = mem(av); b = mem(bv)
        r = Memory{UInt64}(undef, m + n)
        mul_basecase!(r, 0, a, 0, m, b, 0, n)
        @test toref(r, 0, m + n) == toref(a, 0, m) * toref(b, 0, n)
    end
end

@testset "shift/div1 kernels" begin
    rng = MersenneTwister(7)
    for n in (1, 3, 9), cnt in (1, 13, 63), trial in 1:20
        a = mem(rand(rng, UInt64, n)); aref = toref(a, 0, n)
        r = Memory{UInt64}(undef, n)
        hi = lshift!(r, 0, a, 0, n, cnt)
        @test (big(hi) << (64n)) | toref(r, 0, n) == aref << cnt
        lo = rshift!(r, 0, a, 0, n, cnt)
        @test toref(r, 0, n) == aref >> cnt
        @test big(lo) >> (64 - cnt) == aref & ((big(1) << cnt) - 1)
        # in-place
        b = mem(collect(a)); lshift!(b, 0, b, 0, n, cnt)
        @test toref(b, 0, n) == (aref << cnt) & ((big(1) << (64n)) - 1)
        d = rand(rng, UInt64) | 1
        q = Memory{UInt64}(undef, n)
        rem = divrem_1!(q, 0, a, 0, n, d)
        @test toref(q, 0, n) == aref ÷ d && rem == aref % d
    end
end

@testset "invert_limb / div_2by1" begin
    rng = MersenneTwister(11)
    β = big(1) << 64
    for d in UInt64[UInt64(1) << 63, typemax(UInt64), 0x8000000000000001,
                    rand(rng, UInt64, 20) .| (UInt64(1) << 63)...]
        v = invert_limb(d)
        @test big(v) == (β^2 - 1) ÷ d - β
        for trial in 1:50
            u1 = rand(rng, UInt64) % d      # remainder invariant: u1 < d
            u0 = rand(rng, UInt64)
            qq, rr = div_2by1(u1, u0, d, v)
            num = (big(u1) << 64) | u0
            @test big(qq) == num ÷ d && big(rr) == num % d
        end
        # adversarial: numerator just below (d:0) and near quotient boundary
        for (u1, u0) in ((d - 1, typemax(UInt64)), (d - 1, UInt64(0)),
                         (UInt64(0), d - 1), (UInt64(0), d), (d >> 1, d - 1))
            u1 < d || continue
            qq, rr = div_2by1(u1, u0, d, v)
            num = (big(u1) << 64) | u0
            @test big(qq) == num ÷ d && big(rr) == num % d
        end
    end
end

@testset "invert_pi1 / div_3by2" begin
    rng = MersenneTwister(17)
    β = big(1) << 64
    hi = UInt64(1) << 63
    dpairs = Tuple{UInt64,UInt64}[
        (hi, UInt64(0)), (hi, UInt64(1)), (hi, typemax(UInt64)),
        (typemax(UInt64), typemax(UInt64)), (typemax(UInt64), UInt64(0)),
        (hi | UInt64(1), typemax(UInt64) - 1),
    ]
    for _ in 1:30
        push!(dpairs, (rand(rng, UInt64) | hi, rand(rng, UInt64)))
    end
    for (d1, d0) in dpairs
        D = (big(d1) << 64) | d0
        v = invert_pi1(d1, d0)
        @test big(v) == (β^3 - 1) ÷ D - β
        for trial in 1:40
            # numerator top two limbs strictly below ⟨d1,d0⟩
            num = rand(rng, big(0):(D << 64) - 1)
            u2 = UInt64((num >> 128) & typemax(UInt64))
            u1 = UInt64((num >> 64) & typemax(UInt64))
            u0 = UInt64(num & typemax(UInt64))
            (big(u2) << 64) | u1 < D || continue
            q, r1, r0 = div_3by2(u2, u1, u0, d1, d0, v)
            @test big(q) == num ÷ D
            @test (big(r1) << 64) | r0 == num % D
        end
        # boundary numerators: just below D*B, exact multiples, tiny
        for num in ((D << 64) - 1, D * 7, D, big(0), D - 1, (D << 64) - D)
            u2 = UInt64((num >> 128) & typemax(UInt64))
            u1 = UInt64((num >> 64) & typemax(UInt64))
            u0 = UInt64(num & typemax(UInt64))
            (big(u2) << 64) | u1 < D || continue
            q, r1, r0 = div_3by2(u2, u1, u0, d1, d0, v)
            @test big(q) == num ÷ D
            @test (big(r1) << 64) | r0 == num % D
        end
    end
end

@testset "divrem_bc! rare branches" begin
    # These paths fire with probability ~2^-64 on random inputs, so construct
    # them explicitly and run the basecase directly on normalized divisors.
    rng = MersenneTwister(29)
    hib = UInt64(1) << 63

    function checkbc(uref::BigInt, nn, dref::BigInt, m)
        u = Memory{UInt64}(undef, nn)
        for i in 1:nn
            u[i] = UInt64((uref >> (64 * (i - 1))) & typemax(UInt64))
        end
        d = Memory{UInt64}(undef, m)
        for i in 1:m
            d[i] = UInt64((dref >> (64 * (i - 1))) & typemax(UInt64))
        end
        q = Memory{UInt64}(undef, nn - m)
        v = invert_pi1(d[m], d[m-1])
        qh = divrem_bc!(q, 0, u, 0, nn, d, 0, m, v)
        got_q = (big(qh) << (64 * (nn - m))) | toref(q, 0, nn - m)
        @test got_q == uref ÷ dref
        @test toref(u, 0, m) == uref % dref
    end

    β = big(1) << 64
    for trial in 1:100
        # add-back: window = qe·⟨d1,d0⟩·B + w with w < qe·dl0 forces the 3/2
        # estimate one too high once the low-limb borrow lands
        d1 = rand(rng, UInt64) | hib
        d0 = rand(rng, UInt64)
        dl0 = rand(rng, UInt64) | 1
        qe = rand(rng, UInt64(2):typemax(UInt64))
        w = rand(rng, big(0):min(big(qe) * dl0, β) - 1)
        dref = ((big(d1) << 64 | d0) << 64) | dl0
        uref = (big(qe) * (big(d1) << 64 | d0)) << 64 | w
        @assert uref < dref * β && uref ÷ dref == qe - 1
        checkbc(uref, 4, dref, 3)

        # qhat == β-1 special case: top two window limbs equal ⟨d1,d0⟩
        x = rand(rng, UInt64(0):(dl0 - 1))
        uref2 = (((big(d1) << 64 | d0) << 64) | x) << 64 | rand(rng, UInt64)
        @assert uref2 < dref * β
        checkbc(uref2, 4, dref, 3)

        # same two shapes with extra low divisor limbs (m = 5)
        low = rand(rng, big(0):(big(1) << 128) - 1)
        dref5 = (((big(d1) << 64 | d0) << 64) | dl0) << 128 | low
        uref5 = uref * (β^2) + rand(rng, big(0):β^2 - 1)
        uref5 ÷ dref5 < β && checkbc(uref5, 6, dref5, 5)
        uref6 = uref2 * (β^2) + rand(rng, big(0):β^2 - 1)
        uref6 < dref5 * β && checkbc(uref6, 6, dref5, 5)
    end
end

@testset "divrem_1! adversarial divisors" begin
    rng = MersenneTwister(13)
    for n in (1, 2, 5, 17), trial in 1:30
        av = rand(rng, UInt64, n)
        rand(rng) < 0.3 && (av .= typemax(UInt64))
        rand(rng) < 0.3 && (av[n] = UInt64(1))
        a = mem(av); aref = toref(a, 0, n)
        for d in UInt64[1, 2, 3, 10, UInt64(1) << 32, UInt64(1) << 63,
                        typemax(UInt64), typemax(UInt64) - 1, rand(rng, UInt64) | 1]
            q = Memory{UInt64}(undef, n)
            rem = divrem_1!(q, 0, a, 0, n, d)
            @test toref(q, 0, n) == aref ÷ d
            @test rem == aref % d
        end
        # exact division and quotient == 0 cases
        d = rand(rng, UInt64) >> rand(rng, 0:60) + UInt64(1)
        q = Memory{UInt64}(undef, n)
        prod = (aref ÷ d) * d
        if prod > 0
            pv = [UInt64((prod >> (64 * (i - 1))) & typemax(UInt64)) for i in 1:n]
            p = mem(pv)
            rem = divrem_1!(q, 0, p, 0, n, d)
            @test rem == 0 && toref(q, 0, n) == prod ÷ d
        end
    end
end

@testset "divrem_2!" begin
    using NativeBigInt: divrem_2!
    rng = MersenneTwister(22)
    for n in (2, 3, 4, 7, 16, 33), trial in 1:30
        a = mem(rand(rng, UInt64, n))
        # mix of normalized and unnormalized 2-limb divisors, incl. adversarial
        d1 = trial % 3 == 0 ? rand(rng, UInt64) | (UInt64(1) << 63) :
             trial % 3 == 1 ? rand(rng, UInt64) >> rand(rng, 0:62) :
             (UInt64(1) << rand(rng, 0:63))
        d1 == 0 && (d1 = UInt64(1))
        d0 = trial % 5 == 0 ? typemax(UInt64) : rand(rng, UInt64)
        q = Memory{UInt64}(undef, n - 1)
        r1, r0 = divrem_2!(q, 0, a, 0, n, d1, d0)
        aref = toref(a, 0, n)
        dref = (big(d1) << 64) | d0
        @test toref(q, 0, n - 1) == aref ÷ dref
        @test (big(r1) << 64) | r0 == aref % dref
    end
    # in-place: q aliasing a at the same offset
    a = mem(UInt64[typemax(UInt64), typemax(UInt64), typemax(UInt64)])
    aref = toref(a, 0, 3)
    d1, d0 = UInt64(3), typemax(UInt64)
    dref = (big(d1) << 64) | d0
    r1, r0 = divrem_2!(a, 0, a, 0, 3, d1, d0)
    @test toref(a, 0, 2) == aref ÷ dref && (big(r1) << 64) | r0 == aref % dref
end

@testset "submul_2!" begin
    using NativeBigInt: submul_2!
    rng = MersenneTwister(33)
    for n in (1, 2, 3, 7, 8, 9, 16, 31), trial in 1:20
        a = mem(rand(rng, UInt64, n))
        r = mem(rand(rng, UInt64, n))
        trial % 4 == 0 && fill!(r, typemax(UInt64))
        trial % 5 == 0 && fill!(a, typemax(UInt64))
        trial % 7 == 0 && fill!(r, UInt64(0))
        b0, b1 = rand(rng, UInt64), rand(rng, UInt64)
        trial % 3 == 0 && (b0 = b1 = typemax(UInt64))
        rref, aref = toref(r, 0, n), toref(a, 0, n)
        bref = (big(b1) << 64) | b0
        co1, co0 = submul_2!(r, 0, a, 0, n, b0, b1)
        co = (big(co1) << 64) | co0
        # exact identity: result - co·β^n == r - a*b
        @test toref(r, 0, n) - (co << (64n)) == rref - aref * bref
    end
end

@testset "divrem_bc! two-limb qhat add-back" begin
    # d1 = 2^63 (minimal normalized) with saturated low limbs makes the
    # 4/2 single-super-digit estimate exceed the true beta^2 quotient digit
    # often (error up to 2), exercising the add-back correction.
    rng = MersenneTwister(44)
    err1 = 0; err2 = 0
    for m in (3, 4, 5, 8), trial in 1:50
        d = mem(fill(typemax(UInt64), m))
        d[m] = UInt64(1) << 63
        d[m-1] = trial % 2 == 0 ? UInt64(0) : rand(rng, UInt64)
        n = m + rand(rng, 2:6)
        a = mem(rand(rng, UInt64, n))
        dref, aref = toref(d, 0, m), toref(a, 0, n)
        # count super-digit estimate errors via BigInt simulation of the
        # radix-beta^2 schoolbook loop (super-rows from j=qn, scalar at j==1)
        dhat = (big(d[m]) << 64) | d[m-1]
        qn = n + 1 - m
        R = aref
        R >> (64qn) >= dref && (R -= dref << (64qn))
        j = qn
        while j >= 2
            W = R >> (64 * (j - 2))
            top4 = W >> (64 * (m - 2))
            if top4 >> 128 == dhat   # implementation takes a scalar special row
                Ws = R >> (64 * (j - 1))
                R -= (Ws ÷ dref) * dref << (64 * (j - 1))
                j -= 1
                continue
            end
            qhat = top4 ÷ dhat
            qtrue = W ÷ dref
            e = Int(qhat - qtrue)
            e >= 1 && (err1 += 1)
            e >= 2 && (err2 += 1)
            R -= qtrue * dref << (64 * (j - 2))
            j -= 2
        end
        q = Memory{UInt64}(undef, n - m + 1)
        r = Memory{UInt64}(undef, m)
        NativeBigInt.divrem!(q, 0, r, 0, a, 0, n, d, 0, m)
        @test toref(q, 0, n - m + 1) == aref ÷ dref
        @test toref(r, 0, m) == aref % dref
    end
    # the adversarial shape must actually hit the estimate-error path
    @test err1 > 0
end

@testset "submul_2! SIMD cold path" begin
    # r == a*b (mod beta^n) makes every difference limb 0..3, forcing the
    # cold scalar-propagate block in each SIMD iteration.
    using NativeBigInt: submul_2!
    rng = MersenneTwister(55)
    for n in (8, 16, 24, 31), off in 0:3
        a = mem(rand(rng, UInt64, n))
        b0, b1 = rand(rng, UInt64), rand(rng, UInt64)
        aref = toref(a, 0, n)
        bref = (big(b1) << 64) | b0
        prod = (aref * bref) & ((big(1) << (64n)) - 1)
        r = mem(UInt64[UInt64((prod >> (64k)) & typemax(UInt64)) for k in 0:n-1])
        off > 0 && (r[1] = r[1] + UInt64(off))  # results 0..3 in low limb
        rref = toref(r, 0, n)
        co1, co0 = submul_2!(r, 0, a, 0, n, b0, b1)
        co = (big(co1) << 64) | co0
        @test toref(r, 0, n) - (co << (64n)) == rref - aref * bref
    end
end

@testset "divrem_1! normalized two-limb path" begin
    # shapes that stress the lazy-remainder pi2 loop: saturated dividends
    # (max unreduced remainder growth), quotient limbs near typemax
    # (quotient-window carry saturation), boundary divisors, odd/even n
    rng = MersenneTwister(66)
    β = big(1) << 64
    ds = UInt64[UInt64(1) << 63, (UInt64(1) << 63) + 1, typemax(UInt64),
                typemax(UInt64) - 1, UInt64(10)^19, (UInt64(1) << 63) | 1]
    for d in ds, n in (1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 33)
        for trial in 1:8
            av = rand(rng, UInt64, n)
            trial == 1 && fill!(av, typemax(UInt64))
            trial == 2 && (av = [fill(typemax(UInt64), n - 1); d - 1])   # top < d
            if trial == 3   # engineered near-typemax quotient limbs
                qv = fill(typemax(UInt64), n)
                prod = toref(mem(qv), 0, n) * d + (d - 1)
                prod >= β^n && (prod = (β^n - 1) ÷ d * d + (d - 1))
                prod >= β^n && (prod = β^n - 1)
                av = [UInt64((prod >> (64k)) & typemax(UInt64)) for k in 0:n-1]
            end
            a = mem(av)
            q = Memory{UInt64}(undef, n)
            rem = divrem_1!(q, 0, a, 0, n, d)
            aref = toref(a, 0, n)
            @test toref(q, 0, n) == aref ÷ d
            @test rem == aref % d
        end
    end
end

@testset "divrem! small-quotient subtraction path" begin
    rng = MersenneTwister(77)
    for m in (3, 4, 8, 31), k in (0, 1, 2, 3, 7, 8, 9), trial in 1:10
        d = mem(rand(rng, UInt64, m))
        d[m] |= trial % 2 == 0 ? UInt64(1) << 63 : UInt64(1) << 40  # norm + unnorm
        dref = toref(d, 0, m)
        rref = trial % 3 == 0 ? dref - 1 : (trial % 3 == 1 ? big(0) : rand(rng, big(0):dref-1))
        aref = k * dref + rref
        nbits = ndigits(aref, base = 2)
        n = max(cld(nbits, 64), m)   # a needs >= m limbs for the contract
        n * 64 < nbits && (n += 1)
        a = mem(UInt64[UInt64((aref >> (64j)) & typemax(UInt64)) for j in 0:n-1])
        q = Memory{UInt64}(undef, n - m + 1)
        r = Memory{UInt64}(undef, m)
        NativeBigInt.divrem!(q, 0, r, 0, a, 0, n, d, 0, m)
        @test toref(q, 0, n - m + 1) == k
        @test toref(r, 0, m) == rref
    end
end

@testset "mont_ninv / redc!" begin
    using NativeBigInt: mont_ninv, redc!
    rng = MersenneTwister(0xedc)
    for trial in 1:200
        m0 = rand(rng, UInt64) | 1
        @test m0 * mont_ninv(m0) == -UInt64(1)
    end
    for k in (1, 2, 3, 8, 20), trial in 1:20
        m = mem(rand(rng, UInt64, k))
        m[1] |= 1
        m[k] == 0 && (m[k] = 1)
        mref = toref(m, 0, k)
        # T uniform in [0, m*β^k): x*y with x,y < m
        x = mod(rand(rng, big(0):big(2)^(64k)), mref)
        y = mod(rand(rng, big(0):big(2)^(64k)), mref)
        T = x * y
        t = Memory{UInt64}(undef, 2k + 1)
        for i in 1:2k+1
            t[i] = UInt64((T >> (64 * (i - 1))) & typemax(UInt64))
        end
        r = Memory{UInt64}(undef, k)
        redc!(r, 0, t, 0, m, 0, k, mont_ninv(m[1]))
        @test toref(r, 0, k) == mod(T * invmod(big(2)^(64k), mref), mref)
    end
end

@testset "sqrtrem!" begin
    using NativeBigInt: sqrtrem!, sqrt!
    rng = MersenneTwister(0x59b)
    for k in (1, 2, 3, 5, 10, 33), trial in 1:20
        n = 2k
        a = mem(rand(rng, UInt64, n))
        a[n] |= UInt64(1) << 62   # normalization: top limb >= 2^62
        aref = toref(a, 0, n)
        h = k
        s = Memory{UInt64}(undef, h)
        scratch = Memory{UInt64}(undef, NativeBigInt.sqrt_scratch_len(h))
        rhi = sqrtrem!(s, 0, a, 0, n, scratch)
        sref = toref(s, 0, h)
        rref = (big(rhi) << (64h)) | toref(a, 0, h)
        @test sref == isqrt(aref)
        @test rref == aref - sref^2
    end
end

@testset "sqrt! root-only" begin
    using NativeBigInt: sqrt!
    rng = MersenneTwister(0x54b)
    function checksqrt(aref::BigInt, n::Int)
        a = Memory{UInt64}(undef, n)
        for i in 1:n
            a[i] = UInt64(aref >> (64 * (i - 1)) & typemax(UInt64))
        end
        h = n >> 1
        s = Memory{UInt64}(undef, h)
        scratch = Memory{UInt64}(undef, NativeBigInt.sqrt_scratch_len(h))
        sqrt!(s, 0, a, 0, n, scratch)
        @test toref(s, 0, h) == isqrt(aref)
    end
    for k in (2, 3, 5, 10, 33), trial in 1:30
        n = 2k
        # random normalized
        aref = rand(rng, big(2)^(64n - 2):big(2)^(64n) - 1)
        checksqrt(aref, n)
        # near-perfect squares: t² - 1 forces the root correction, t² and
        # t² + small keep R tiny so the cheap positivity checks can't fire
        t = rand(rng, big(2)^(32n - 1):big(2)^(32n) - 1)
        for d in (big(-1), big(0), big(1), 2t - 1, 2t)
            v = t^2 + d
            big(2)^(64n - 2) <= v < big(2)^(64n) && checksqrt(v, n)
        end
    end
end

@testset "divrem_bc! approx-mode quotient" begin
    using NativeBigInt: divrem_bc!
    rng = MersenneTwister(0xd1a)
    hib = UInt64(1) << 63
    β = big(1) << 64

    # Returns q̂ - floor(u/d); the contract is one-sided over-approximation
    # by at most 2 ulps (spec §2.2/§2.4), with u destroyed and no remainder.
    function apprerr(uref::BigInt, nn, dref::BigInt, m)
        u = Memory{UInt64}(undef, nn)
        for i in 1:nn
            u[i] = UInt64((uref >> (64 * (i - 1))) & typemax(UInt64))
        end
        d = Memory{UInt64}(undef, m)
        for i in 1:m
            d[i] = UInt64((dref >> (64 * (i - 1))) & typemax(UInt64))
        end
        q = Memory{UInt64}(undef, nn - m)
        v = invert_pi1(d[m], d[m-1])
        qh = divrem_bc!(q, 0, u, 0, nn, d, 0, m, v, true)
        got_q = (big(qh) << (64 * (nn - m))) | toref(q, 0, nn - m)
        return got_q - uref ÷ dref
    end

    maxerr = big(0)
    for m in (2, 3, 4, 5, 8, 13, 40, 97), qn in (1, 2, 3, m - 1, m, m + 2)
        qn < 1 && continue
        nn = m + qn
        for trial in 1:15
            dref = rand(rng, big(0):β^m - 1) | (big(hib) << (64 * (m - 1)))
            uref = rand(rng, big(0):β^nn - 1)
            err = apprerr(uref, nn, dref, m)
            @test 0 <= err <= 2
            maxerr = max(maxerr, err)
        end
        # adversarial: minimal top limb with all-ones low limbs maximizes the
        # mass a truncating implementation neglects; boundary numerators sit
        # where a 1-ulp overestimate flips the floor
        dref = (big(hib) << (64 * (m - 1))) | (β^(m - 1) - 1)
        for trial in 1:15
            k = rand(rng, big(0):β^(qn - 1) - 1) << 64 | rand(rng, UInt64)
            k = min(k, β^qn - 1)
            for w in (big(0), big(1), dref - 1, dref ÷ 2)
                uref = k * dref + w
                uref < β^nn || continue
                err = apprerr(uref, nn, dref, m)
                @test 0 <= err <= 2
                maxerr = max(maxerr, err)
            end
        end
    end
    @test maxerr <= 2
end

@testset "sqrt! divappr certificate path" begin
    using NativeBigInt: sqrt!, SQRT_DIVAPPR_THRESHOLD
    rng = MersenneTwister(0x5dca)
    function checksqrt(aref::BigInt, n::Int)
        a = Memory{UInt64}(undef, n)
        for i in 1:n
            a[i] = UInt64(aref >> (64 * (i - 1)) & typemax(UInt64))
        end
        h = n >> 1
        s = Memory{UInt64}(undef, h)
        scratch = Memory{UInt64}(undef, NativeBigInt.sqrt_scratch_len(h))
        sqrt!(s, 0, a, 0, n, scratch)
        @test toref(s, 0, h) == isqrt(aref)
    end
    # sizes with l = h>>1 straddling and well above the divappr gate
    thr = SQRT_DIVAPPR_THRESHOLD
    for h in (2thr - 2, 2thr - 1, 2thr, 2thr + 1, 2thr + 5, 3thr, 512)
        n = 2h
        for trial in 1:4
            checksqrt(rand(rng, big(2)^(64n - 2):big(2)^(64n) - 1), n)
        end
        # adversarial: perfect squares sit on the certificate boundary
        # (always-fallback), t²±1 exercise correction and tiny remainders,
        # t²-t forces the confident-decrement branch, 2t-ish maxes R
        t = rand(rng, big(2)^(32n - 1):big(2)^(32n) - 1)
        for d in (-t, big(-1), big(0), big(1), t, 2t - 1, 2t)
            v = t^2 + d
            big(2)^(64n - 2) <= v < big(2)^(64n) && checksqrt(v, n)
        end
    end
end
