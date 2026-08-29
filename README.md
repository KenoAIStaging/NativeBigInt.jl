# NativeBigInt.jl

Pure-Julia arbitrary-precision integer type (`NBig`) targeting GMP-competitive
performance from ~100 bits up. Requires a recent Julia (uses `Memory{UInt64}`).
Headline results (details and tables in Benchmarks below):

- `+`/`-`/`*`/`divrem`: faster than GMP through ~4k bits; `*` pulls ahead
  again from ~10k bits as the fp NTT takes over, growing to 2–4× from
  ~35k bits up.
- `gcd`/`gcdx`: within ~25% of GMP across the range, ahead from ~16k bits
  (subquadratic HGCD).
- `powermod`: parity around ~16k bits, ahead above (NTT-backed Barrett
  reduction).
- `isqrt`: within ~15% of GMP across the range, ahead below ~512 bits and
  above ~32k.

## Algorithms

Three layers, mirroring GMP's mpn/mpz split: raw limb kernels, sign-free
mpn-style algorithms over limb buffers, and the `NBig` value type on top.
Dispatch thresholds are benchmark-tuned (`bench/bench_kernels.jl`).

- **Kernels (`src/kernels/`):** sign-free limb-vector primitives —
  `add_n!`/`sub_n!`, `mul_1!`/`addmul_1!`/`submul_1!`, `mul_2!`/`addmul_2!`,
  `mul_basecase!`, `lshift!`/`rshift!`, `cmp_limbs`. Division uses a
  reciprocal-based `divrem_1!` (Möller–Granlund 2/1 normalized inverse, with
  a fused-shift path for unnormalized divisors), `div_3by2` for quotient
  estimation, and `divrem_bc!` — a radix-β² Knuth Algorithm D that produces
  two quotient limbs per pass via chained `div_3by2` estimates and a SIMD
  `submul_2!`, with a small-quotient subtraction fast path for near-equal
  bit lengths. Kernels are written so LLVM emits `adc`/`mulx` chains, with
  SIMD.jl fast paths and scalar cold paths for carry-chaining edge cases.

- **Multiplication (`src/mul.jl`):** subtractive Karatsuba (threshold ~29
  limbs) with a general unbalanced-operand path; `mul!`/`sqr!` hand off to
  the fp NTT at ~124 (mul) / 160 (sqr) balanced limbs. The NTT benchmarks
  better than Toom-3 at every size (and thus presumably better than the
  higher-degree Toom variants), so no Toom layer exists. There is no short-
  product (Mulders) layer either: `mullo!`/`sqrlo!` existed but every
  potential caller sits above the NTT crossover, where a truncated product
  saves nothing, so they were removed (see git history).

- **fp NTT multiplication (`src/fpntt.jl`):** the sole large-size engine —
  number-theoretic transforms computed entirely in `Float64` in the style of
  FLINT's `fft_small`, with the convolution recombined by CRT over two
  just-under-2^50 primes (p₁ = 65205·2^34 + 1, p₂ = 2^50 − 2^38 + 1, both
  with 3²·5·7 | p−1; the ≈2^99.99 working modulus keeps the chunk width at
  ~46 bits at typical sizes, Garner recombination in the unpack). Products
  use the FMA error-free transform, reduction is a Barrett quotient via the
  magic-constant round, and twiddles are stored as Shoup-style `(w, w/p)`
  pairs — every operation is exactly correct by rounding-error analysis, not
  approximation, with lazy per-butterfly bounds keeping the hot loop pure
  mul/fma/add traffic (the analysis lives at the top of `src/fpntt.jl`).
  Radix-4 DIF forward; the inverse is the transposed forward network (same
  tables, reverse stage order — no inverse twiddles or 1/N pre-scale pass).
  Transform lengths are m·2^k for m | 315 (Winograd radix-3/5/7 passes,
  radix 3 up to twice) with bench-placed per-multiplier admission floors, so
  zero-padding waste is ≤ ~12.5% at small sizes and ≤ ~7% once transforms go
  DRAM-bound. Squaring runs one forward transform instead of two.

- **Division (`src/div.jl`):** multi-limb `divrem!` — Knuth Algorithm D over
  `divrem_bc!` below ~100 limbs, GMP-`dcpi1`-style divide-and-conquer
  division above (recursive 2n/n blocks over `mul!`, so it inherits
  Karatsuba and the NTT; 0.82–0.96× GMP's `mpn_tdiv_qr` from the crossover
  through at least 2048 limbs). The divisor and quotient sides are gated
  separately: `DC_DIV_THRESHOLD` on the divisor, the lower
  `DC_DIV_PARTIAL_THRESHOLD` on the quotient, since the leading partial
  block's cross-product arm beats schoolbook as soon as `mul!` is
  subquadratic. Sharing one threshold cost up to 2.1× on short quotients over
  large divisors. All four tier/mode combinations run through one
  quotient-peeling driver, `div_blocks!`, parametrized by engine (`DcEngine`
  here, Barrett `MuEngine` over `invert.jl`'s reciprocal) and by an
  exact-vs-approximate bottom block. The approximate mode (driven directly by
  `sqrt.jl`, the only consumer) is a one-sided quotient (never below the true
  one, within ~20 ulps dc / 6 mu) that skips all remainder work, via a
  triangle-truncated schoolbook mode (`divrem_bc!` with `approx`) and a dc
  recursion that peels the top quotient half exactly.

- **Barrett division (`src/invert.jl`):** `invertappr!` is a multi-limb
  approximate reciprocal — the Newton-doubling generalization of the
  `invert_limb`/`invert_pi1` kernels (GMP `mpn_invertappr`), one-sided and
  within 1 ulp, recursing above `INV_NEWTON_THRESHOLD = 72` and otherwise
  taking one direct 2n/n division. `mu_divrem!` (GMP `mpn_mu_div_qr`) builds
  blocked Barrett division on it: two multiplications per quotient block
  instead of dc division's recursive division per block, having paid for one
  reciprocal up front. That makes it O(M(n)) against dc's O(M(n)·log n),
  which is why `divrem!` dispatches to it once the reciprocal amortizes —
  measured D/M rises 4.3 → 6.1 → 7.8 at 512/2048/8192 limbs. The crossover
  moves with the *shape*, since the reciprocal is spread over
  `k = ⌈qn/m⌉` blocks: 0.91× dc for a balanced 2m/m at 1024 limbs, 0.90× for
  3m/m at 384, 0.87× for 5m/m at 256, improving to 0.40–0.68× by 4096.

- **gcd (`src/gcd.jl`):** Lehmer gcd/gcdext (Knuth Algorithm L) on 126-bit
  leading windows — two bracket-verified single-word phases per window,
  one fused matrix pass over the operands, and for gcdx a V-cofactor pair
  carried in lockstep. Above ~300 limbs (gcd; ~250 for gcdx) a subquadratic
  HGCD layer in the style of GMP's `mpn_hgcd` (Möller 2008) takes over:
  recursive half-gcd builds a 2×2 matrix of det-+1 mpn cofactors whose
  products route through `mul!`, for an O(M(n) log n) total — ahead of both
  the Lehmer loop and GMP's `mpn_gcd` from ~600 limbs (1.2× at 2000 limbs,
  widening).

- **Higher-level algorithms (`src/algorithms.jl`):**
  - *sqrt:* Karatsuba square root (Zimmermann), each level dividing the
    halved numerator by the top of the root buffer in place (no divisor
    construction or per-level renormalization). For `isqrt` above ~4k bits
    the top level is root-only: the divappr engines with one guard limb plus a
    mantissa-interval certificate settles the root without computing the
    remainder or the final square, reconstructing them only in the ambiguous
    band (~2⁻⁵⁷ of inputs, plus perfect squares).
  - *powermod:* sliding-window exponentiation reducing each product with
    Montgomery `redc!` (`src/montgomery.jl`, odd moduli) or `divrem!` (even)
    at small sizes, and above per-parity thresholds (~68 limbs odd, ~240
    even) a plain-domain Barrett reduction (HAC 14.42, `src/barrett.jl`)
    whose two per-product multiplies ride `mul!`.
  - *radix conversion* for `string`/`parse`: power-of-two bases pack/unpack
    bit windows directly (O(n)); other bases use per-limb `divrem_1!` /
    Horner below `STR_DC_THRESHOLD` (40 limbs) and a recursive
    split/combine around a `bb^(2^i)` power tree above, for O(M(n) log n)
    (the recursion lives in `src/nbig.jl`).

- **`NBig` (`src/nbig.jl`):** sign-magnitude value type
  (`signlen = sign * limb count`, little-endian normalized `Memory{UInt64}`),
  covering comparison, `+`/`-`/`*`, `divrem`/`div`/`rem`/`mod`/`fld`/`cld`,
  shifts, two's-complement bitwise ops, `^`, base 2–36 string conversion,
  and conversions to/from `Int64`/`UInt64`/`Int128`/`BigInt`/`Float64`.

## Benchmarks

`bench/bench_highlevel.jl` compares `NBig` against `Base.BigInt` end-to-end
(construction not included) across the target bit-size regime. Ratio is
NBig time / BigInt time — lower is better; ≤1.0 means NBig is faster.
Shapes: `divrem` is 2n/n; `gcd`/`gcdx` operands share a planted n/2-bit
factor; `powermod` uses an n-bit modulus and a 512-bit-capped exponent.

| op | 128 | 256 | 512 | 1k | 2k | 4k | 8k | 16k | 32k |
|---|---|---|---|---|---|---|---|---|---|
| `+` | 0.51 | 0.55 | 0.92 | 0.60 | 0.88 | 0.75 | 1.02 | 1.48 | 1.21 |
| `-` | 0.53 | 0.50 | 0.89 | 0.60 | 0.83 | 0.88 | 1.17 | 1.82 | 1.55 |
| `*` | 0.50 | 0.61 | 0.87 | 0.87 | 1.12 | 1.19 | 1.12 | 0.84 | 0.65 |
| `divrem` | 0.41 | 0.37 | 0.66 | 0.86 | 1.05 | 0.82 | 0.88 | 1.04 | 1.10 |
| `gcd` | 1.04 | 1.11 | 1.09 | 1.08 | 1.16 | 1.09 | 1.06 | 0.99 | 0.81 |
| `gcdx` | 1.01 | 0.99 | 1.25 | 1.25 | 1.25 | 1.22 | 1.12 | 0.97 | 0.79 |
| `isqrt` | 0.81 | 0.96 | 1.14 | 1.09 | 1.12 | 1.14 | 1.07 | 1.05 | 0.94 |
| `powermod` (odd) | 2.00 | 1.14 | 1.95 | 1.67 | 1.36 | 1.33 | 1.48 | 1.07 | 0.76 |
| `powermod` (even) | 2.05 | 2.31 | 1.98 | 1.82 | 1.15 | 1.11 | 1.36 | 1.06 | 0.77 |

The broad shape: the core ring ops (`+`/`-`/`*`/`divrem`) beat GMP through
~4k bits and `*` pulls ahead again from ~10k as the fp NTT takes over
(`+`/`-` drift behind at 8k+, where both sides are memory-bound and GMP's
in-place reallocation wins). `gcd`/`gcdx` sit within ~25% throughout and
lead from ~16k. `powermod` pays GMP's assembly-Montgomery tax at small
sizes, reaches parity around 16k bits, and leads by ~25% at 32k where
Barrett reduction rides the NTT. `isqrt` sits within ~15% of GMP across the
range (per-input variance ±15%): the root-only divappr top level plus the
normalized-divisor level step and pool-sized allocations closed the former
~40% mid-range gap. Decimal `string`/`parse` are supported and benchmarked
in `bench_highlevel.jl` but slower than GMP's.

At the kernel level, Karatsuba carries ~2k–8k bits at rough GMP parity
(0.95–1.09× against `__gmpn_mul`, trading blows with GMP's hand-tuned Toom
assembly), and the two-prime fp NTT takes over at ~8k bits already ahead
(`bench/bench_kernels.jl mul`, AVX-512 machine; ratio is `mul!` /
`__gmpn_mul` on two equal operands):

| bits | 10k  | 16k  | 33k  | 49k  | 66k  | 98k  | 131k | 262k | 524k | 2.1M | 16.8M | 268M |
|------|------|------|------|------|------|------|------|------|------|------|-------|------|
| `*`  | 0.90 | 0.69 | 0.50 | 0.46 | 0.37 | 0.34 | 0.29 | 0.29 | 0.24 | 0.27 | 0.27  | 0.43 |

The lead reaches 2–4× from ~35k bits up, including the range where GMP has
switched to its own Schönhage–Strassen FFT. Squaring crosses over at the
same point with a similar lead. The fixed-prime chunk width shrinks slowly
as operands grow (the ratio drifting up in the 268M-bit column), but the
two-prime working modulus keeps the engine ahead through the largest sizes
that fit in memory.

Kernel- and threshold-level benchmarks against GMP's `__gmpn_*` functions
live in `bench/bench_kernels.jl`, a family-selecting driver (`micro`,
`mul`, `sqr`, `div`, `kar`, `dc`, `gcd`, `barrett`, `sqrt`); run
it with no args for the family list.
