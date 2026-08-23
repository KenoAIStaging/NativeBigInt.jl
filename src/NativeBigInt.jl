module NativeBigInt

import SIMD

export NBig

const Limb = UInt64
const DLimb = UInt128
const V8 = SIMD.Vec{8,Limb}   # SIMD width shared by the add/sub and mul kernels

include("kernels/addsub.jl")
include("kernels/mul.jl")
include("kernels/shift.jl")
include("kernels/div.jl")
include("montgomery.jl")
include("mul.jl")
include("div.jl")
include("invert.jl")
include("sqrt.jl")
include("algorithms.jl")
include("gcd.jl")
include("nbig.jl")
include("fpntt.jl")

end
