using Distributions, StatsFuns, StatsPlots, ProgressMeter
include("forest_task.jl")
include("plot_funs.jl")

## Simulation of the forest environment
nIter = 10_000
nSteps = 50
darkZoneRadius = [0.2, 0.4, 0.6]
stepsize = 0.1
s = 1.0

# Model parameters
σ² = 1.0 # Perceptual variance
β = 5.0 # Inverse temperature 
a0 = 2 # Initial confidence 
λ = 0.2 # Leak 
θ = 10.0 # Slope of the commitment probability 
τ = 0.6 # Offset of the commitment probability (~threshold)
ξ = 0.0 # Inference noise


# Run models
models = [:full, :partial]

belief, firstpass, commited, confidence = forest_task_simu(nSteps, nIter; darkZoneRadius = darkZoneRadius, stepsize = stepsize, s = s, σ² = σ², ξ = ξ, θ = θ, τ = τ, β = β, λ = λ, a0 = a0, mdl = models);

_, Dfirstpass1, Dcommited1, Dconfidence1 = forest_task_simu(nSteps, nIter; darkZoneRadius = darkZoneRadius, stepsize = stepsize, s = s, σ² = σ², ξ = ξ, θ = θ, τ = τ, β = β, λ = 0.25, a0 = a0, mdl = [:partial]);

_, Dfirstpass2, Dcommited2, Dconfidence2 = forest_task_simu(nSteps, nIter; darkZoneRadius = darkZoneRadius, stepsize = stepsize, s = s, σ² = σ², ξ = ξ, θ = θ, τ = τ, β = β, λ = 0.3, a0 = a0, mdl = [:partial]);

## Individual traces 
indBelief = zeros(nSteps)
indCommit = zeros(nSteps)
indConfidence = zeros(nSteps)
trace = zeros(nSteps-1)
latent = initialize_full_model(a0)
forest_task!(indBelief, zeros(nSteps), indCommit, indConfidence, latent, trace; mdl = :partial, darkZoneRadius = 0.4, stepsize = stepsize, s = s, σ² = σ², ξ = ξ, θ = θ, τ = τ, β = β, λ = λ, ρ = 1/2/pi)

plot(trace, ylims=(-1, 1), label="", linewidth=3, color=:black)
plot!(zeros(nSteps-1), ribbon = fill(0.4, nSteps-1), color=:grey, linewidth=0, xticks=[], yticks=[], xlabel="Steps", ylabel="Position", size=(500, 500), dpi=300, background_color=:transparent, label="")


## Plots
# First pass 
noCommit = selectdim(firstpass, 3, 1)
noCommitFP_pl = nice_bar_plot(noCommit; title="Full bayesian model", legend_title="Radius of uninformative zone", label= reshape(darkZoneRadius,1,:))

withCommit = selectdim(firstpass, 3, 2)
withCommitFP_pl = nice_bar_plot(withCommit; title="Commitment model", legend_title="Radius of uninformative zone", label= reshape(darkZoneRadius,1,:))

extCommit = dropdims(Dfirstpass1, dims=3)
extCommitFP1_pl = nice_bar_plot(extCommit; title="Commitment model", legend_title="Radius of uninformative zone", label= reshape(darkZoneRadius,1,:))

extCommit = dropdims(Dfirstpass2, dims=3)
extCommitFP2_pl = nice_bar_plot(extCommit; title="Commitment model", legend_title="Radius of uninformative zone", label= reshape(darkZoneRadius,1,:))

# Belief
noCommit = selectdim(belief, 3, 1)
noCommitBelief_pl = nice_plot(noCommit;title="Full bayesian model", xlabel="Steps", ylabel="Belief", ylims = (0,1), label = reshape(darkZoneRadius,1,:), legend_title="Radius of uninformative zone", legend_position= :topright)

withCommit = selectdim(belief, 3, 2)
withCommitBelief_pl = nice_plot(withCommit;title="Commitment model", xlabel="Steps", ylabel="Belief", ylims = (0,1), label = reshape(darkZoneRadius,1,:), legend_title="Radius of uninformative zone", legend_position= :topright)

# Proba commitment
lowCommit = squeeze_mean(commited[:,:,2]; dims=2)
extCommit1 = squeeze_mean(Dcommited1[:,:,1]; dims=2)
extCommit2 = squeeze_mean(Dcommited2[:,:,1]; dims=2)
Pcommit_pl = nice_plot(hcat(lowCommit, extCommit1, extCommit2);title="", xlabel="Steps", ylabel="Prob. maintained commitment", ylims = (0,1))



