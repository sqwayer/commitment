using Distributions, StatsFuns, StatsPlots, ProgressMeter
include("sequential_task.jl")
include("plot_funs.jl")

## Simulation of the beads task
nIter = 10_000
nSteps = 15
process = Bernoulli
s = 1.5

# Model parameters
σ² = Inf # Perceptual variance (Beads task : Inf)
κ = 0.9 # Decision criterion in beads task (between 0.5 and 1)
a0 = 4 # Initial confidence 
λ = 0.2 # Leak 
θ = 5.0 # Slope of the commitment probability 
τ = 0.95 # Offset of the commitment probability (~threshold)
ξ = 0.0 # Inference noise
ρ = 0.5

# Run models
models = [:full, :partial]

belief, firstpass, commited, uncertainty = seq_task_simu(nSteps, nIter; process = process, s = s, σ² = σ², ξ = ξ, θ = θ, τ = τ, λ = λ, β = 1.0, a0 = a0, ρ = ρ, κ = κ, mdl = models);

Dbelief, Dfirstpass, Dcommited, Duncertainty = seq_task_simu(nSteps, nIter; process = process, s = s, σ² = σ², ξ = ξ, θ = θ, τ = τ, λ = 0.3, β = 1.0, a0 = a0, ρ = ρ, κ = κ, mdl = [:partial]);

nice_bar_plot(hcat(firstpass, Dfirstpass), xlims=(0, 15.5), xlabel="# Draws to decision", labels=["Full Bayesian model" "Commitment model λ = 0.2" "Commitment model λ = 0.3"], ylims=(0, 0.2))


## Simulations of the sequential evidence integration task
nIter = 10_000
nSteps = 10
process = VonMises
s = 2.0

# Model parameters
σ² = 2.0 # Perceptual variance (Beads task : Inf)
β = 5.0 # Inverse temperature 
κ = 0.9 # Decision criterion
a0 = 6 # Initial confidence 
λ = 0.25 # Leak 
θ = 30.0 # Slope of the commitment probability 
τ = 0.15 # Offset of the commitment probability (~threshold)
ξ = 0.0 # Inference noise

# Run models on individual traces to split commited vs non commited
#belief = zeros(nSteps, nIter)
#uncertainty = zeros(nSteps, nIter)
#commit = zeros(nSteps, nIter)

finalcommited = zeros(nIter)
finaluncertainty = zeros(nIter)
lambdas = zeros(nIter)
@Threads.threads for i = 1:nIter
    lamb = rand([0.1, 0.2, 0.3])
    epLength = rand(5:25)

    latent = initialize_full_model(a0)
    latent[:lpfun] = vonmises_llh
    commit = zeros(epLength)
    firstpass = zeros(epLength)

    seq_task!(zeros(epLength), firstpass, commit, zeros(epLength), latent; mdl = :partial, epLength=epLength, lesionInt=[0], process=VonMises, σ²=σ², s=s, τ=τ, β=β, θ=θ, ξ=0.0, λ=lamb, λ₂=0.0, ρ = 1/2/pi, κ =κ)
    lambdas[i] = lamb
    finalcommited[i] = mean(commit)
    finaluncertainty[i] =  latent[:Uf]
end
scatter(finaluncertainty, finalcommited, zcolor=lambdas, label="", colormap = cgrad(:algae, categorical=true), clims=(0.0, 0.3), msw=0, colorbar=false, xlabel="Final uncertainty", ylabel="% Discounted evidence", background_color=:transparent, size=(500, 500), dpi=300)
