using Distributions, StatsFuns, StatsPlots, ProgressMeter, DataFrames
include("../../environments/sequential_task.jl")
include("../plot_funs.jl")


"""
Simulations of a beads task and an continuous evidence integration task for figure 4
The model observes stimuli passively, we monitor belief and uncertainty
"""
## Simulation of the beads task
nIter = 1_000_000
nSteps = 15
process = Bernoulli
s = 1.5

# Model parameters
σ² = Inf # Perceptual variance (Beads task : Inf)
κ = 0.9 # Decision criterion in beads task (between 0.5 and 1)
a0 = 4 # Initial confidence 
λ = 0.1 # Leak 
θ = 5.0 # Slope of the commitment probability 
τ = 0.95 # Offset of the commitment probability (~threshold)
ξ = 0.0 # Inference noise
ρ = 0.5

# Run models
models = [:full, :partial]

belief, firstpass, commited, uncertainty = seq_task_simu(nSteps, nIter; process = process, s = s, σ² = σ², ξ = ξ, θ = θ, τ = τ, λ = λ, λ₂ = λ, β = 1.0, a0 = a0, ρ = ρ, κ = κ, mdl = models);

Dbelief, Dfirstpass, Dcommited, Duncertainty = seq_task_simu(nSteps, nIter; process = process, s = s, σ² = σ², ξ = ξ, θ = θ, τ = τ, λ = 0.25, λ₂ = 0.2, β = 1.0, a0 = a0, ρ = ρ, κ = κ, mdl = [:partial]);

nice_plot(hcat(firstpass, Dfirstpass), palfn = PTolsGrad2, linewidth=5, xlims=(0, 15), xlabel="# Draws to decision", yticks=(0.0:0.05:0.2, 0:5:20), ylabel="Proportion (%)", ylims=(0, 0.2), legendfontsize=14)

## Simulations of the sequential evidence integration task
nIter = 1_000_000
process = VonMises
s = 2.0
θ = 100.0 # Slope of the commitment probability 
τ = 0.5 # Offset of the commitment probability (~threshold)

# Model parameters
σ² = 2.0 # Perceptual variance (Beads task : Inf)

# Run models on individual traces to split commited vs non commited
finalcommited = zeros(nIter)
discounted = zeros(nIter)
finaluncertainty = zeros(nIter)
lambdas = zeros(nIter)
@Threads.threads for i = 1:nIter
    lamb =  0.0 + 0.25*rand() 
    epLength = rand(5:25)

    latent = initialize_full_model(a0)
    latent[:lpfun] = vonmises_llh
    commit = zeros(epLength)
    firstpass = zeros(epLength)

    seq_task!(zeros(epLength), firstpass, commit, zeros(epLength), latent; mdl = :partial, epLength=epLength, lesionInt=[0], process=VonMises, σ²=σ², s=s, τ=τ, β=β, θ=θ, ξ=0.0, λ=lamb, λ₂ = lamb, ρ = 1/2/pi, κ =κ)
    lambdas[i] = lamb
    discounted[i] = mean(commit)#any(commit .== 1) ? findfirst(commit .== 1) : 0.0#
    finalcommited[i] = latent[:commited]
    finaluncertainty[i] = latent[:Uf]
end
## Plot heatmap
idx = findall(0.1 .<= lambdas .<= 0.25) #Weak leak interval: 0.0-0.15, Strong leak interval: 0.1-0.25
nice_heatmap(finaluncertainty[idx], discounted[idx], range(0.05, 0.35, length=40), range(0, 1.0, length=20), ylabel="Biased evidence integration\n(% observations)", xlabel="Final uncertainty", yticks=(0.2:0.2:0.8, 20:20:80), xticks=(0.1:0.1:0.3), colorbar=nothing)


