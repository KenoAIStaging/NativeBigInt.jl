# Local-only kernel-selection driver: pick which family to sweep, optionally
# override the sizes. Never commit results — headline numbers go in the
# commit message. Run one family alone; a concurrent bench process corrupts
# the timings.
#
#   julia --startup-file=no --project=. bench/bench_kernels.jl <family> [sizes...]
#
# families:
#   micro    raw row kernels vs GMP mpn (add_n, mul_1, addmul_1/2, basecase, shifts)
#   mul      mul! vs __gmpn_mul (+ kara column above the NTT threshold)
#   sqr      sqr! vs __gmpn_sqr, plus basecase/kar and kar/NTT crossovers
#   div      divrem! / divrem_1! vs __gmpn_tdiv_qr / __gmpn_divrem_1
#   kar      KARATSUBA_THRESHOLD sweep (mul_kar! vs gmp)
#   dc       DC_DIV_THRESHOLD sweep (divrem_dc!/bc! vs gmp)
#   gcd      subquadratic gcd/gcdx threshold sweep (Lehmer vs DC vs gmp)
#   mullo    mullo!/sqrlo! sweep vs the full product they replace
#   barrett  BARRETT_THRESHOLD / BARRETT_EVEN_THRESHOLD sweep (powermod_limbs)
#   sqrt     SQRT_DIVAPPR_THRESHOLD sweep (isqrt divappr vs exact vs gmp)
#   radix    fp NTT fwd+rev ns/point per odd multiplier m (tunes ntt_len's
#            family admission: a multiplier only deserves selection where its
#            per-point cost stays below the padding it saves)
#
# Trailing integer args override the size list of the family they follow.
using NativeBigInt, BenchmarkTools, Random
using NativeBigInt: Limb,
    add_n!, sub_n!, mul_1!, addmul_1!, addmul_2!, submul_1!, mul_basecase!,
    lshift!, rshift!, mul!, sqr!, mul_kar!, sqr_basecase!, sqr_kar!,
    sqr_fpntt2!, kar_scratch_len, sqr_scratch_len, MUL_FPNTT_THRESHOLD,
    divrem!, divrem_1!, divrem_dc!, divrem_bc!, invert_pi1, gcd!, gcdext!,
    mullo!, sqrlo!, mullo_basecase!, sqrlo_basecase!, mullo_scratch_len,
    sqrlo_scratch_len, powermod_limbs,
    fp_ntt_plan, fp_ntt_fwd!, fp_ntt_rev!, FP_CTX1, fp_prime, two_adicity

# large sizes are µs–ms scale; a modest per-measurement budget keeps sweeps quick
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.5

# ---- GMP mpn references ----------------------------------------------------
g_add!(r, a, b, n)  = ccall((:__gmpn_add_n, :libgmp), Limb, (Ptr{Limb}, Ptr{Limb}, Ptr{Limb}, Clong), r, a, b, n)
g_sub!(r, a, b, n)  = ccall((:__gmpn_sub_n, :libgmp), Limb, (Ptr{Limb}, Ptr{Limb}, Ptr{Limb}, Clong), r, a, b, n)
g_mul1!(r, a, n, x) = ccall((:__gmpn_mul_1, :libgmp), Limb, (Ptr{Limb}, Ptr{Limb}, Clong, Limb), r, a, n, x)
g_am1!(r, a, n, x)  = ccall((:__gmpn_addmul_1, :libgmp), Limb, (Ptr{Limb}, Ptr{Limb}, Clong, Limb), r, a, n, x)
g_am2!(r, a, n, bp) = ccall((:__gmpn_addmul_2, :libgmp), Limb, (Ptr{Limb}, Ptr{Limb}, Clong, Ptr{Limb}), r, a, n, bp)
g_sm1!(r, a, n, x)  = ccall((:__gmpn_submul_1, :libgmp), Limb, (Ptr{Limb}, Ptr{Limb}, Clong, Limb), r, a, n, x)
g_mbc!(r, a, m, b, n) = ccall((:__gmpn_mul_basecase, :libgmp), Cvoid, (Ptr{Limb}, Ptr{Limb}, Clong, Ptr{Limb}, Clong), r, a, m, b, n)
g_lsh!(r, a, n, c)  = ccall((:__gmpn_lshift, :libgmp), Limb, (Ptr{Limb}, Ptr{Limb}, Clong, Cuint), r, a, n, c)
g_rsh!(r, a, n, c)  = ccall((:__gmpn_rshift, :libgmp), Limb, (Ptr{Limb}, Ptr{Limb}, Clong, Cuint), r, a, n, c)
g_mul!(r, a, m, b, n) = ccall((:__gmpn_mul, :libgmp), Limb, (Ptr{Limb}, Ptr{Limb}, Clong, Ptr{Limb}, Clong), r, a, m, b, n)
g_sqr!(r, a, n)     = ccall((:__gmpn_sqr, :libgmp), Cvoid, (Ptr{Limb}, Ptr{Limb}, Clong), r, a, n)
g_tdiv!(q, r, a, nn, d, m) = ccall((:__gmpn_tdiv_qr, :libgmp), Cvoid, (Ptr{Limb}, Ptr{Limb}, Clong, Ptr{Limb}, Clong, Ptr{Limb}, Clong), q, r, 0, a, nn, d, m)
g_divrem1!(q, a, n, d) = ccall((:__gmpn_divrem_1, :libgmp), Limb, (Ptr{Limb}, Clong, Ptr{Limb}, Clong, Limb), q, 0, a, n, d)
g_gcd!(gp, up, un, vp, vn) = ccall((:__gmpn_gcd, :libgmp), Clong, (Ptr{Limb}, Ptr{Limb}, Clong, Ptr{Limb}, Clong), gp, up, un, vp, vn)

# Generic size sweep. `cands(n)` returns [name => thunk], each thunk returning
# its own measured seconds; the first is the reference the rest are ratioed
# against. Thunks own their setup, so destructive kernels copy per-eval and
# stateful ones (global thresholds) can flip state before timing.
function sweep(sizes, cands; unit = 1e9, u = "ns")
    for n in sizes
        cs = cands(n)
        ref = cs[1].second()
        print(rpad("$n", 8), rpad("$(cs[1].first)=$(round(ref * unit, digits = 1))$u", 24))
        for c in @view cs[2:end]
            t = c.second()
            print(rpad("$(c.first)=$(round(t * unit, digits = 1))$u($(round(t / ref, digits = 2)))", 28))
        end
        println(); flush(stdout)
    end
end

rlimbs(n) = Memory{Limb}(rand(Limb, n))

# ---- families --------------------------------------------------------------
# micro: each row kernel (nbig) against its own GMP counterpart (the reference).
function run_micro(sizes)
    kernels = [
        "add_n"    => (n -> (a = rlimbs(n); b = rlimbs(n); r = Memory{Limb}(undef, n + 1);
            ["gmp" => (() -> @belapsed g_add!($r, $a, $b, $n)), "nbig" => (() -> @belapsed add_n!($r, 0, $a, 0, $b, 0, $n))])),
        "sub_n"    => (n -> (a = rlimbs(n); b = rlimbs(n); r = Memory{Limb}(undef, n + 1);
            ["gmp" => (() -> @belapsed g_sub!($r, $a, $b, $n)), "nbig" => (() -> @belapsed sub_n!($r, 0, $a, 0, $b, 0, $n))])),
        "mul_1"    => (n -> (a = rlimbs(n); x = rand(Limb); r = Memory{Limb}(undef, n + 1);
            ["gmp" => (() -> @belapsed g_mul1!($r, $a, $n, $x)), "nbig" => (() -> @belapsed mul_1!($r, 0, $a, 0, $n, $x))])),
        "addmul_1" => (n -> (a = rlimbs(n); x = rand(Limb); r = Memory{Limb}(rand(Limb, n + 1));
            ["gmp" => (() -> @belapsed g_am1!($r, $a, $n, $x)), "nbig" => (() -> @belapsed addmul_1!($r, 0, $a, 0, $n, $x))])),
        "addmul_2" => (n -> (a = rlimbs(n); bp = rlimbs(2); r = Memory{Limb}(rand(Limb, n + 2));
            ["gmp" => (() -> @belapsed g_am2!($r, $a, $n, $bp)), "nbig" => (() -> @belapsed addmul_2!($r, 0, $a, 0, $n, $(bp[1]), $(bp[2])))])),
        "submul_1" => (n -> (a = rlimbs(n); x = rand(Limb); r = Memory{Limb}(rand(Limb, n + 1));
            ["gmp" => (() -> @belapsed g_sm1!($r, $a, $n, $x)), "nbig" => (() -> @belapsed submul_1!($r, 0, $a, 0, $n, $x))])),
        "basecase" => (n -> (a = rlimbs(n); b = rlimbs(n); r = Memory{Limb}(undef, 2n);
            ["gmp" => (() -> @belapsed g_mbc!($r, $a, $n, $b, $n)), "nbig" => (() -> @belapsed mul_basecase!($r, 0, $a, 0, $n, $b, 0, $n))])),
        "lshift"   => (n -> (a = rlimbs(n); r = Memory{Limb}(undef, n + 1);
            ["gmp" => (() -> @belapsed g_lsh!($r, $a, $n, 13)), "nbig" => (() -> @belapsed lshift!($r, 0, $a, 0, $n, 13))])),
        "rshift"   => (n -> (a = rlimbs(n); r = Memory{Limb}(undef, n);
            ["gmp" => (() -> @belapsed g_rsh!($r, $a, $n, 13)), "nbig" => (() -> @belapsed rshift!($r, 0, $a, 0, $n, 13))])),
    ]
    for (name, cands) in kernels
        println(name, " (nbig ratio vs gmp):"); sweep(sizes, cands)
    end
end

run_mul(sizes) = sweep(sizes, n -> begin
    a = rlimbs(n); b = rlimbs(n); r = Memory{Limb}(undef, 2n)
    cs = ["gmp" => (() -> @belapsed g_mul!($r, $a, $n, $b, $n)),
          "mul!" => (() -> @belapsed mul!($r, 0, $a, 0, $n, $b, 0, $n))]
    if n >= MUL_FPNTT_THRESHOLD           # kara column times the path the NTT replaced
        s = Memory{Limb}(undef, kar_scratch_len(n))
        push!(cs, "kar" => (() -> @belapsed mul_kar!($r, 0, $a, 0, $b, 0, $n, $s, 0)))
    end
    cs
end)

function run_sqr(sizes)
    println("sqr! / mul!(a,a) vs gmp:")
    sweep(sizes, n -> begin
        a = rlimbs(n); r = Memory{Limb}(undef, 2n)
        ["gmp" => (() -> @belapsed g_sqr!($r, $a, $n)),
         "sqr!" => (() -> @belapsed sqr!($r, 0, $a, 0, $n)),
         "mul!(a,a)" => (() -> @belapsed mul!($r, 0, $a, 0, $n, $a, 0, $n))]
    end)
    # crossover sweeps reuse `sizes`, each filtered to its relevant range
    bc_sizes = filter(n -> 16 <= n <= 96, sizes)
    isempty(bc_sizes) || println("\ncrossover: basecase vs one-level Karatsuba")
    sweep(bc_sizes, n -> begin
        a = rlimbs(n); r = Memory{Limb}(undef, 2n)
        s = Memory{Limb}(undef, kar_scratch_len(n, n))   # force one kar level
        ["bc" => (() -> @belapsed sqr_basecase!($r, 0, $a, 0, $n)),
         "kar" => (() -> @belapsed sqr_kar!($r, 0, $a, 0, $n, $s, 0, $n))]
    end)
    ntt_sizes = filter(n -> n >= 96, sizes)
    isempty(ntt_sizes) || println("\ncrossover: Karatsuba vs fp NTT")
    sweep(ntt_sizes, n -> begin
        a = rlimbs(n); r = Memory{Limb}(undef, 2n)
        # the scratch malloc is timed, not hoisted: sqr! allocates it per call,
        # and excluding it flatters Karatsuba enough to move the threshold
        ["kar" => (() -> @belapsed sqr_kar!($r, 0, $a, 0, $n,
                                            Memory{Limb}(undef, sqr_scratch_len($n)), 0)),
         "ntt" => (() -> @belapsed sqr_fpntt2!($r, 0, $a, 0, $n))]
    end; unit = 1e6, u = "us")
end

function run_div(sizes)
    println("divrem_1! vs mpn_divrem_1 (n limbs / 1):")
    sweep(sizes, n -> begin
        a = rlimbs(n); d = rand(Limb) | 1; q = Memory{Limb}(undef, n)
        ["gmp" => (() -> @belapsed g_divrem1!($q, $a, $n, $d)),
         "divrem_1!" => (() -> @belapsed divrem_1!($q, 0, $a, 0, $n, $d))]
    end)
    println("\ndivrem! vs mpn_tdiv_qr (normalized, then unnormalized divisor):")
    shapes = ((4, 2), (8, 4), (16, 8), (32, 16), (64, 32), (128, 64),
              (16, 2), (64, 4), (128, 8), (64, 60), (128, 120))
    divrem_row(norm) = ((n, m)) -> begin
        a = rlimbs(n); d = rlimbs(m)
        d[end] = norm ? (d[end] | Limb(1) << 63) : ((d[end] >> 17) | 1)  # top bit set / clear
        q = Memory{Limb}(undef, n - m + 1); r = Memory{Limb}(undef, m)
        ["gmp" => (() -> @belapsed g_tdiv!($q, $r, $a, $n, $d, $m)),
         "divrem!" => (() -> @belapsed divrem!($q, 0, $r, 0, $a, 0, $n, $d, 0, $m))]
    end
    sweep(shapes, divrem_row(true))
    println("(unnormalized)")
    sweep(((16, 8), (64, 32), (128, 64)), divrem_row(false))
end

run_kar(sizes) = sweep(sizes, n -> begin
    thrs = (25, 29, 33, 37, 41, 49, 57, 65)
    a = rlimbs(n); b = rlimbs(n); r = Memory{Limb}(undef, 2n)
    s = Memory{Limb}(undef, kar_scratch_len(n, minimum(thrs)))  # smallest thr needs the most
    vcat(["gmp" => (() -> @belapsed g_mul!($r, $a, $n, $b, $n))],
         ["T=$t" => (() -> @belapsed mul_kar!($r, 0, $a, 0, $b, 0, $n, $s, 0, $t)) for t in thrs])
end)

run_dc(sizes) = sweep(sizes, m -> begin
    thrs = (30, 40, 50, 65, 80, 110)
    nn = 2m; d = rlimbs(m); d[m] |= Limb(1) << 63             # normalized divisor
    u = rlimbs(nn); v = invert_pi1(d[m], d[m-1])
    q = Memory{Limb}(undef, nn - m + 1); r = Memory{Limb}(undef, m)
    # divrem_dc!/bc! destroy the numerator, so each sample re-copies it
    vcat(["gmp" => (() -> @belapsed g_tdiv!($q, $r, $u, $nn, $d, $m)),
          "bc" => (() -> @belapsed divrem_bc!($q, 0, uu, 0, $nn, $d, 0, $m, $v) setup = (uu = copy($u)) evals = 1)],
         ["T=$t" => (() -> @belapsed divrem_dc!($q, 0, uu, 0, $nn, $d, 0, $m, $v, $t) setup = (uu = copy($u)) evals = 1) for t in thrs])
end; unit = 1e6, u = "us")

function run_gcd(sizes)
    isempty(sizes) && (sizes = (100, 150, 200, 300, 400, 600, 900, 1400, 2000, 3000))
    randmag(n) = (a = rlimbs(n); a[n] |= Limb(1) << 63; a)
    BIG = typemax(Int) ÷ 2
    println("gcd: Lehmer vs DC(dc,hgcd) vs GMP")
    sweep(sizes, n -> begin
        a = randmag(n); b = randmag(n); b[1] |= one(Limb); g = Memory{Limb}(undef, n)
        cs = ["lehmer" => (() -> @belapsed (u = copy($a); v = copy($b); gcd!(u, $n, v, $n; dc_thr = $BIG, hgcd_thr = $BIG)) evals = 1),
              "gmp" => (() -> @belapsed (u = copy($a); v = copy($b); g_gcd!($g, u, $n, v, $n)) evals = 1)]
        for (dc, hg) in ((150, 100), (250, 100), (400, 130), (600, 130))
            dc > n && continue
            push!(cs, "dc$dc/hg$hg" => (() -> @belapsed (u = copy($a); v = copy($b); gcd!(u, $n, v, $n; dc_thr = $dc, hgcd_thr = $hg)) evals = 1))
        end
        cs
    end; unit = 1e6, u = "us")
    println("\ngcdext: Lehmer vs DC")
    sweep(filter(<=(2000), sizes), n -> begin
        a = randmag(n); b = randmag(n)
        ext(dc, hg) = () -> @belapsed (u = Memory{Limb}(undef, $n + 3); v = Memory{Limb}(undef, $n + 3);
            copyto!(u, 1, $a, 1, $n); copyto!(v, 1, $b, 1, $n); gcdext!(u, $n, v, $n; dc_thr = $dc, hgcd_thr = $hg)) evals = 1
        cs = ["lehmer" => ext(BIG, BIG)]
        for (dc, hg) in ((120, 100), (200, 100), (300, 130), (500, 130))
            dc > n && continue
            push!(cs, "dc$dc/hg$hg" => ext(dc, hg))
        end
        cs
    end; unit = 1e6, u = "us")
end

function run_mullo(sizes)
    println("mullo!/mullo_basecase! vs full mul!:")
    sweep(sizes, k -> begin
        a = rlimbs(k); b = rlimbs(k); r = Memory{Limb}(undef, 2k + 2)
        ms = Memory{Limb}(undef, max(1, mullo_scratch_len(k)))
        cs = ["mul!" => (() -> @belapsed mul!($r, 0, $a, 0, $k, $b, 0, $k)),
              "mullo!" => (() -> @belapsed mullo!($r, 0, $a, 0, $k, $b, 0, $k, $k, $ms, 0))]
        k <= 320 && push!(cs, "bc" => (() -> @belapsed mullo_basecase!($r, 0, $a, 0, $k, $b, 0, $k, $k)))
        cs
    end)
    println("\nsqrlo!/sqrlo_basecase! vs full sqr!:")
    sweep(sizes, k -> begin
        a = rlimbs(k); r = Memory{Limb}(undef, 2k + 2)
        ss = Memory{Limb}(undef, max(1, sqrlo_scratch_len(k)))
        cs = ["sqr!" => (() -> @belapsed sqr!($r, 0, $a, 0, $k)),
              "sqrlo!" => (() -> @belapsed sqrlo!($r, 0, $a, 0, $k, $k, $ss, 0))]
        k <= 320 && push!(cs, "bc" => (() -> @belapsed sqrlo_basecase!($r, 0, $a, 0, $k, $k)))
        cs
    end)
end

# barrett: powermod_limbs with Barrett forced off (cur, the reference) vs on
# (bar), per parity. Fixed 512-bit exponent keeps the mul/reduce count per k
# constant, so the bar/cur ratio isolates the reduction cost.
function run_barrett(sizes)
    e = rand(big(2)^511:big(2)^512 - 1)
    for (label, par) in (("odd", 1), ("even", 0))
        println(label, " modulus (bar ratio vs cur):")
        sweep(sizes, k -> begin
            m = rlimbs(k); m[k] |= Limb(1) << 62               # top limb well away from 0
            m[1] = (m[1] & ~Limb(1)) | Limb(par)
            b = rlimbs(k); b[k] = m[k] >> 1                    # guarantees b < m
            ["cur" => (() -> @belapsed powermod_limbs($b, $k, $e, $m, $k, false) seconds = 2),
             "bar" => (() -> @belapsed powermod_limbs($b, $k, $e, $m, $k, true) seconds = 2)]
        end; unit = 1e6, u = "us")
    end
end

# sqrt: SQRT_DIVAPPR_THRESHOLD must be a mutable typed global while tuning.
function run_sqrt(bits)
    isempty(bits) && (bits = (2048, 4096, 8192, 16384, 32768, 65536))
    sweep(bits, b -> begin
        ab = rand(big(2)^(b - 1):big(2)^b - 1); an = NBig(ab)
        thrs = (4, 8, 16, 24, 32, 48, 64, 96, 128)
        setthr(t) = () -> (NativeBigInt.SQRT_DIVAPPR_THRESHOLD = t; @belapsed isqrt($an))
        vcat(["gmp" => (() -> @belapsed isqrt($ab)),
              "off" => setthr(typemax(Int) >> 1)],
             ["thr=$t" => setthr(t) for t in thrs])
    end; unit = 1e6, u = "us")
end

# radix: per-point fwd+rev transform cost by odd multiplier m, at point-count
# targets T (one block per T; N = m·2^k is the smallest m-length >= T).  The
# ratio column is vs the pure pow2 transform of the same coverage — an m is
# worth admitting in ntt_len where its ratio is below the padding factor it
# saves (up to 2·(next multiplier gap)).
function run_radix(targets)
    F = FP_CTX1
    p = fp_prime(F)
    maxk = two_adicity(F)
    for T in targets
        println("T = $T points:")
        ref = 0.0
        for m in (1, 3, 5, 7, 9, 15, 21, 35, 45, 63, 105, 315)
            k = max(Base.top_set_bit(cld(T, m) - 1), 2)
            k <= maxk || continue
            N = m << k
            x = Float64.(rand(UInt64(0):p-1, N))
            plan = fp_ntt_plan(N, F)
            t = @belapsed (fp_ntt_fwd!($x, $plan); fp_ntt_rev!($x, $plan))
            perpt = t / N * 1e9
            m == 1 && (ref = perpt)
            println("  ", rpad("m=$m", 7), rpad("N=$N", 12),
                    rpad("$(round(perpt, digits = 3)) ns/pt", 14),
                    "(", round(perpt / ref, digits = 2), ")")
            flush(stdout)
        end
    end
end

# ---- dispatch --------------------------------------------------------------
const FAMILIES = Dict(
    "micro"   => (run_micro,   [1, 2, 4, 8, 16, 32, 64, 128, 256]),
    "radix"   => (run_radix,   [1 << k for k in 8:16]),
    "mul"     => (run_mul,     [8, 16, 25, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048]),
    "sqr"     => (run_sqr,     [2, 4, 8, 12, 16, 24, 32, 40, 48, 64, 96, 128, 160, 192, 256, 384, 512]),
    "div"     => (run_div,     [2, 4, 8, 16, 32, 64, 128]),
    "kar"     => (run_kar,     [32, 48, 64, 80, 96, 128, 192, 256]),
    "dc"      => (run_dc,      [32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048]),
    "gcd"     => (run_gcd,     Int[]),
    "mullo"   => (run_mullo,   [16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 256, 320, 384, 448, 512, 640, 768]),
    "barrett" => (run_barrett, [4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 64, 80, 96, 128, 192, 256]),
    "sqrt"    => (run_sqrt,    Int[]),
)

function main(args)
    if isempty(args) || !haskey(FAMILIES, args[1])
        println("usage: bench_kernels.jl <family> [sizes...]")
        println("families: ", join(sort(collect(keys(FAMILIES))), ", "))
        return
    end
    fn, default_sizes = FAMILIES[args[1]]
    sizes = length(args) > 1 ? parse.(Int, args[2:end]) : default_sizes
    fn(sizes)
end

main(ARGS)
