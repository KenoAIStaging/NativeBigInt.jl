# Local benchmark of the O(n (log n)^(1-2^-78)) multiplier against mul!
# (fp NTT) and GMP, with the paper's parameter formulas at each size.  Prints
# a markdown table; the ratio column is BelowNLogN / mul! (lower is better,
# and it is never lower than about 10^4).
#
#   julia --startup-file=no --project=. bench/bench_belownlogn.jl [bits...]
#
# Trailing integer args override the size list.  Pass `d=<k>` to force k
# axes (the paper's formula d = ⌊⌈log2 n⌉^(1/40)⌋ gives 1 below 2^(2^40) bits).
using NativeBigInt, Random
using NativeBigInt: Limb, mul!
const BN = NativeBigInt.BelowNLogN

args = copy(ARGS)
dpos = findfirst(a -> startswith(a, "d="), args)
d_override = dpos === nothing ? nothing : parse(Int, args[dpos][3:end])
dpos === nothing || deleteat!(args, dpos)
sizes = isempty(args) ? (2^10, 2^12, 2^14, 2^16, 2^18, 2^20) : parse.(Int, args)

g_mul!(r, a, m, b, n) = ccall((:__gmpn_mul, :libgmp), Limb, (Ptr{Limb}, Ptr{Limb}, Clong, Ptr{Limb}, Clong), r, a, m, b, n)

rng = MersenneTwister(0x5ab)
function timeit(f, reps)
    f()
    best = Inf
    for _ in 1:reps
        t = @elapsed f()
        best = min(best, t)
    end
    best
end

println("| bits | d | p | T | setup s | below-n-log-n s | mul! s | GMP s | ratio vs mul! |")
println("|---|---|---|---|---|---|---|---|---|")
for n in sizes
    x = rand(rng, big(2)^(n-1):big(2)^n-1); y = rand(rng, big(2)^(n-1):big(2)^n-1)
    X, Y = NBig(x), NBig(y)
    nl = cld(n, 64)
    prm = BN.Params(n; d=d_override)
    tsetup = @elapsed mp = BN.MulPlan(prm)
    r = Memory{Limb}(undef, 2nl)
    tb = timeit(() -> BN.multiply!(r, 0, X.limbs, 0, nl, Y.limbs, 0, nl, mp), 2)
    BigInt(NativeBigInt.nbig_from_limbs(1, r, 2nl)) == x * y || error("wrong product at n = $n")
    r2 = Memory{Limb}(undef, 2nl)
    tm = timeit(() -> mul!(r2, 0, X.limbs, 0, nl, Y.limbs, 0, nl), 50)
    tg = timeit(() -> g_mul!(r2, X.limbs, nl, Y.limbs, nl), 50)
    println("| $n | $(prm.d) | $(prm.p) | $(prm.T) | $(round(tsetup, digits=2)) | $(round(tb, digits=3)) | $(round(tm, sigdigits=3)) | $(round(tg, sigdigits=3)) | $(round(Int, tb / tm)) |")
end
