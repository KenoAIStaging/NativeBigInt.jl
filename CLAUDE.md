# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

NativeBigInt.jl is a pure-Julia arbitrary-precision integer type (`NBig`) aiming
for GMP-competitive performance in the ~100–4000 bit regime. It reimplements the
mpn/mpz split of GMP in Julia and depends only on SIMD.jl (plus an optional
Random extension). Requires a recent Julia — it uses `Memory{UInt64}`.

## Commands

```bash
# Run a single test file — load the package, then include the file.
# (test/runtests.jl just includes the per-area files under one @testset.)
julia --project -e 'using NativeBigInt, Test, Random; include("test/test_kernels.jl")'

# Benchmarks (not part of CI; run under the bench env, which pins the deps)
julia --project=bench bench/bench_highlevel.jl [op...]        # NBig vs Base.BigInt, end-to-end
julia --project=bench bench/bench_kernels.jl <family> [sizes] # kernel/threshold selection vs GMP
```

`bench/bench_kernels.jl` takes a family (`micro`, `mul`, `sqr`, `div`, `kar`,
`dc`, `gcd`, `barrett`, `sqrt`) plus optional sizes; run it with no
args for the family list. `bench/bench_highlevel.jl` takes optional op filters.

There is no build/lint step — it's a plain Julia package.

## Architecture

**The layer structure (kernels → algorithms → NBig, mirroring GMP's mpn/mpz
split) and the per-layer algorithm inventory are documented in `README.md` —
read its Algorithms section for orientation, and update it there (not here)
when algorithms change.** The module include order in `src/NativeBigInt.jl`
reflects the dependency direction. Agent-relevant notes the README doesn't
cover:

- **Kernels (`src/kernels/*.jl`)** are sign-free primitives over raw
  `Memory{Limb}` (`Limb == UInt64`) with explicit offsets and lengths, written
  so LLVM emits `adc`/`mulx` chains, with SIMD.jl fast paths (`V8` =
  `SIMD.Vec{8,Limb}`, the shared width) and scalar cold paths for
  carry-chaining edge cases. Performance here is the whole point — check
  changes against `bench/` and, where relevant, the generated asm
  (`bench/asm_dump.jl`).
- **Dispatch thresholds live at the mpn layer** (`src/mul.jl`,
  `src/div.jl`, `src/gcd.jl`), never at the NBig level, and are
  benchmark-tuned: the constants live in those files (Karatsuba, fp NTT,
  divide-and-conquer division, subquadratic HGCD gcd) — retune them with
  `bench_kernels.jl kar` / `dc` / `gcd`.
- **Deleted algorithms:** Toom-3, the integer Goldilocks NTT (`src/ntt.jl`),
  and the single-prime fp pipeline were removed once the two-prime fp NTT
  beat them everywhere — git history has them; don't reintroduce variants
  without benchmark cause.
- **`src/belownlogn/`** is the preprint algorithm "Integer multiplication
  below n log n" (OpenAI, 2026-09-23), implemented faithfully in the RAM
  model and exact at every size; `mul!` dispatches to it at the paper's
  real cutoff, n ≥ 2^(6^(2^131)) bits, which `MUL_BELOWNLOGN_THRESHOLD`
  stores as 131 on the log₂∘log₆ scale of ⌈log₂ n⌉ (nothing representable
  reaches it), and it is 10⁴–10⁵× slower than the fp NTT at every size. It is the one engine
  here that exists for correctness and faithfulness, not benchmarks; do not
  "tune" it into something else, and keep its tests (`test_belownlogn.jl`)
  exact against rational/BigFloat references. Include order inside the
  submodule is the dependency order: fixedwidth → f2 → motifs → layers →
  synthetic → resampling → assembly. `bench/bench_belownlogn.jl` times it.
- `ext/NativeBigIntRandomExt.jl` is a weak-dependency extension (loaded when
  `Random` is available) providing `rand` over `NBig` ranges.

## Conventions and invariants

- `signlen` encodes both sign and length; `0` is the unique zero (`limbs` is
  `EMPTY_LIMBS`). Magnitude length is `nlimbs(x) = abs(x.signlen)`.
- Limb buffers are little-endian and **top-limb-normalized** (no leading zero
  limbs) at the NBig boundary; use `normlen`/`nbig_from_limbs` when constructing.
  Kernel buffers may be over-allocated — always pass explicit lengths/offsets.
- Kernels take `(buffer, offset, ..., n)` and are `@inbounds`; the caller owns
  bounds and aliasing correctness.

## Testing structure

Tests are split by layer and by strategy. `test_kernels.jl`, `test_algorithms.jl`,
`test_nbig.jl` are per-layer unit tests. `test_differential.jl` and
`test_mixed.jl` cross-check results against `Base.BigInt` over randomized inputs —
this is the primary correctness net, so new operations should be added to the
differential comparison against `BigInt`.
