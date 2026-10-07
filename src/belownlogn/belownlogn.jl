# Integer multiplication below n log n (OpenAI Math Release preprint
# OAI:Integer-multiplication-below-n-log-n-September-23-2026), implemented in
# the RAM model.  See README.md, section "Multiplication below n log n", for
# what this is and is not.
module BelowNLogN

using ..NativeBigInt: Limb, mul!, normlen, negate_twos!, NBig, nbig_from_limbs

include("fixedwidth.jl")
include("f2.jl")
include("motifs.jl")
include("layers.jl")
include("synthetic.jl")
include("resampling.jl")
include("assembly.jl")

end
