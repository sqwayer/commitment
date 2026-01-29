using Distributions, StatsFuns, StatsPlots, ProgressMeter
include("sequential_task.jl")
include("plot_funs.jl")

## Simulation of the conitnuous sequential evidence integration environment
nIter = 5_000
nSteps = 750
epLength = 30
lesionInt = 210:550
process = VonMises
s = 2.0

# Model parameters
σ² = 2.0 # Perceptual variance
β = 5.0 # Inverse temperature 
a0 = 6 # Initial confidence 
λ = 0.2 # Leak 
λ₂ = 0.4 # Lesioned leak
θ = 15.0 # Slope of the commitment probability 
τ = 0.5 # Offset of the commitment probability (~threshold)
ξ = 0.0 # Inference noise

# Run models
models = [:full, :partial]

belief, fp, commited, uncertainty = seq_task_simu(nSteps, nIter; process = process, epLength = epLength, lesionInt = lesionInt, s = s, σ² = σ², ξ = ξ, θ = θ, τ = τ, β = β, λ = λ, λ₂ = λ₂, a0 = a0, mdl = models);

##
plot(repeat([1, 0], inner=epLength, outer=Int(floor(nSteps / epLength/ 2))), color=:grey, linestyle=:dash, label="True category", ylims=(-0.05, 1.05))
plot!(belief[:,2]; linewidth=3, color=:black, ylabel="", xlabel="Trials", label="Belief Category 1", size=(500, 500), dpi=300, background=:transparent, legend_background_color=:white, legend_position=:bottom)





