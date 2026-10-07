using NativeBigInt, Test, Random

@testset "NativeBigInt" begin
    include("test_kernels.jl")
    include("test_algorithms.jl")
    include("test_nbig.jl")
    include("test_differential.jl")
    include("test_fpntt.jl")
    include("test_mixed.jl")
    include("test_rand.jl")
    include("test_belownlogn.jl")
end
