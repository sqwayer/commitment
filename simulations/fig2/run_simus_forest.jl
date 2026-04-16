using Distributions, StatsFuns, StatsPlots, ProgressMeter
include("../../environments/forest_task.jl")
include("../plot_funs.jl")

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

Dbelief, Dfirstpass2, Dcommited2, Dconfidence2 = forest_task_simu(nSteps, nIter; darkZoneRadius = darkZoneRadius, stepsize = stepsize, s = s, σ² = σ², ξ = ξ, θ = θ, τ = τ, β = β, λ = 0.3, a0 = a0, mdl = [:partial]);

## Individual traces 
nTraces = 10
trace = zeros(nSteps-1, nTraces)
indBelief = zeros(nSteps, nTraces)
for i = 1:nTraces
    indCommit = zeros(nSteps)
    indConfidence = zeros(nSteps)


    latent = initialize_full_model(a0)
    forest_task!(@views(indBelief[:,i]), zeros(nSteps), indCommit, indConfidence, latent, @views(trace[:,i]); mdl = :partial, darkZoneRadius = 0.4, stepsize = stepsize, s = s, σ² = σ², ξ = ξ, θ = θ, τ = τ, β = β, λ = 0.2, ρ = 1/2/pi)
end


plot(zeros(nSteps-1), ribbon = fill(0.4, nSteps-1), color=:grey, linewidth=0, xticks=[], yticks=[], xlabel="Steps", ylabel="Position", size=(500, 250), dpi=300, leftmargin=1cm, background_color=:transparent, label="")
plot!(trace, ylims=(-1, 1), label="", linewidth=3, color=PTolsGrad1(3)[1], alpha=0.3)
plot!(trace[:,1], ylims=(-1, 1), label="", linewidth=5, color=PTolsGrad2(3)[1], alpha=1.0)
hline!([0.9], color=:black, linewidth=5, linestyle=:dot, label="")


## Plots
# First pass 
noCommit = selectdim(firstpass, 3, 1)
noCommitFP_pl = nice_bar_plot(noCommit[:,2]; color=PTols[3], label= "", rightmargin=1cm, ylims=(0, 0.5))

withCommit = selectdim(firstpass, 3, 2)
withCommitFP_pl = nice_bar_plot(withCommit[:,2];  label="", rightmargin=1cm, ylims=(0, 0.5))

extCommit = dropdims(Dfirstpass1, dims=3)
extCommitFP1_pl = nice_bar_plot(extCommit[:,2]; color=PTolsGrad1(3)[2], label= "", rightmargin=1cm, ylims=(0, 0.5))

extCommit = dropdims(Dfirstpass2, dims=3)
extCommitFP2_pl = nice_bar_plot(extCommit[:,2]; color=PTols[4], label= "", rightmargin=1cm, ylims=(0, 0.5))

## Belief
noCommit = selectdim(belief, 3, 1)
noCommitBelief_pl = nice_plot(noCommit;title="Full bayesian model", xlabel="Steps", ylabel="Belief", ylims = (0,1), label = reshape(darkZoneRadius,1,:), legend_title="Radius of uninformative zone", legend_position= :topright)

withCommit = selectdim(belief, 3, 2)
withCommitBelief_pl = nice_plot(withCommit;title="Commitment model", xlabel="Steps", ylabel="Belief", ylims = (0,1), label = reshape(darkZoneRadius,1,:), legend_title="Radius of uninformative zone", legend_position= :topright)

extCommit = dropdims(Dbelief, dims=3)
extCommitBelief_pl = nice_plot(extCommit;title="Commitment model", xlabel="Steps", ylabel="Belief", ylims = (0,1), label = reshape(darkZoneRadius,1,:), legend_title="Radius of uninformative zone", legend_position= :topright)

## Uncertainty
unc = selectdim(confidence[2:end,:,:], 2, 2)
unc = hcat(unc, Dconfidence1[2:end,2,:], Dconfidence2[2:end,2,:])
uncertainty_pl = nice_plot(unc; xlabel="Steps", ylabel="Uncertainty", ylims = (0.2,0.8), label = "", palette=ColorScheme(vcat(PTols[3], range(PTols[1], PTols[4], length=3))), linewidth=5, labelfontsize=14, tickfontsize=14)


# withCommit = selectdim(belief, 3, 2)
# withCommitBelief_pl = nice_plot(withCommit;title="Commitment model", xlabel="Steps", ylabel="Belief", ylims = (0,1), label = reshape(darkZoneRadius,1,:), legend_title="Radius of uninformative zone", legend_position= :topright)

# extCommit = dropdims(Dbelief, dims=3)
# extCommitBelief_pl = nice_plot(extCommit;title="Commitment model", xlabel="Steps", ylabel="Belief", ylims = (0,1), label = reshape(darkZoneRadius,1,:), legend_title="Radius of uninformative zone", legend_position= :topright)
## Proba commitment
lowCommit = squeeze_mean(commited[:,:,2]; dims=2)
extCommit1 = squeeze_mean(Dcommited1[:,:,1]; dims=2)
extCommit2 = squeeze_mean(Dcommited2[:,:,1]; dims=2)
Pcommit_pl = nice_plot(hcat(lowCommit, extCommit1, extCommit2);title="", xlabel="Steps", ylabel="Prob. maintained commitment", ylims = (0,1), linewidth=5, labelfontsize=14, tickfontsize=14)



