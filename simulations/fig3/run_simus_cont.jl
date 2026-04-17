using Distributions, StatsFuns, StatsPlots, ProgressMeter
include("../../environments/sequential_task.jl")
include("../plot_funs.jl")

""" 
Simulations of a perceptual decision making task with a reversal, used for figure 3. 
The agent observes stimuli drawn from on of 2 Von Mises 
distributions and updates its beliefs about the category 
The value of the leak lambda is increased just before a reversal occurs
"""

## Simulation of the conitnuous sequential evidence integration environment
nIter = 5_000
nSteps = 600
epLength = 300
lesionInt = 200:600
process = VonMises
s = [[0.5, 0.5], [1.0, 1.0], [2.0, 2.0]]

# Model parameters
σ² = s # Perceptual variance
β = 5.0 # Inverse temperature 
a0 = 10 # Initial confidence 
λ = 0.05 # Leak 
λ₂ = 0.1 # Lesioned leak
θ = 100.0# Slope of the commitment probability
τ = 0.2 # Offset of the commitment probability (~threshold)
ξ = 0.0 # Inference noise

# Run models
models = [:full, :partial]

belief = zeros(nSteps+1, 2, 3)
uncertainty = zeros(nSteps+1, 2, 3)
for varIdx = 1:3
    belief[:,:,varIdx], fp, commited, uncertainty[:,:,varIdx] = seq_task_simu(nSteps, nIter; process = process, epLength = epLength, lesionInt = lesionInt, s = s[varIdx], σ² = s[varIdx], ξ = ξ, θ = θ, τ = τ, β = β, λ = λ, λ₂ = λ₂, a0 = a0, mdl = models);
end

## Plot belief
plot(belief[100:500,:, 3]; linewidth=5, palette=PTolsGrad2(2), ylabel="Belief cat. 1", xlabel="# Observations", label="", labelfontsize=14, tickfontsize=14, size=(500, 500), dpi=300, background=:transparent)

for vid = [2, 1]
    plot!(belief[100:500,:, vid]; linewidth=5, palette=PTolsGrad2(2), ylabel="Belief cat. 1", xlabel="# Observations", label="", alpha=vid/3)
end

plot!(repeat([1, 0], inner=epLength-100, outer=Int(floor(nSteps / epLength/ 2))), color=:grey, linestyle=:dash, label="", ylims=(-0.05, 1.05))

## Plot uncertainty
idx = 100:500
plot(uncertainty[idx,:, 3]; linewidth=5, palette=PTolsGrad2(2), ylabel="Uncertainy", xlabel="# Observations", ylims=(0.09, 0.22),yticks=0.1:0.02:0.2, label="", labelfontsize=14, tickfontsize=14, size=(500, 500), dpi=300, background=:transparent)

for vid = [2, 1]
    plot!(uncertainty[idx,:, vid]; linewidth=5, palette=PTolsGrad2(2), ylabel="Uncertainy", xlabel="# Observations", label="", alpha=vid/3)
end
plot!()

## Illustrations of VM distributions
X = range(-pi, pi, length=1000)
plot(X, [exp.(vonmises_llh.(pi/2, 2.0, X)), exp.(vonmises_llh.(-pi/2, 2.0, X))], palette=PTolsGrad1(2), linewidth=5, label="", xlabel="Observations", ylabel="Likelihood", xticks=[], yticks=[], labelfontsize=14, size=(500, 500), dpi=300, background=:transparent)

plot!(X, [exp.(vonmises_llh.(pi/2, 1.0, X)), exp.(vonmises_llh.(-pi/2, 1.0, X))], palette=PTolsGrad1(2), linewidth=5, label="", alpha=0.5)

##
plot(X, exp.(vonmises_llh.(0.0, 2.0, X)), ylims=(0,exp(vonmises_llh(0.0, 2.0, 0.0))+0.03), color=PTols[1], linewidth=20, label="", axis=nothing, border=:none, size=(500, 500), dpi=300, background=:transparent)