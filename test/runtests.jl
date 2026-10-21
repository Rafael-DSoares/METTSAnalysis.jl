using METTSAnalysis
using Test

@testset "METTSAnalysis.jl" begin
    include("test_pruning.jl")
    include("test_analysis.jl")
    include("test_io.jl")
    include("test_mbar.jl")
end
