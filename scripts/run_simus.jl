using Distributions, StatsFuns, StatsPlots, ProgressMeter
include("forest_simple.jl")

## Simulation of the simple forest environment
nIter = 10_000
nSteps = 50

inverseTemp = [0.0, 2.0, 4.0, Inf]
perceptualNoise = [0.5, 2.0, 4.0, 6.0]
commitmentThresh = [0.65, 0.7]

belief, firstpass, commited, choiceSwitch = forest_simple(nSteps, nIter; inverseTemp = inverseTemp, perceptualNoise = perceptualNoise, commitmentThresh = commitmentThresh, ν = 0.05, σ² = 1.0, ξ = 0.0);

## Plots
legend_labels = reshape(["β = $b" for b in inverseTemp], 1,:)
## 1a) Perceptual noise vs action variability: belief
noCommit = selectdim(belief, 2, 2) 

noCommitBelief_pl = [nice_plot(noCommit[:,:,i];title="Noise = $(perceptualNoise[i])", xlabel="Steps", ylabel="Belief", ylims = (0,1), label = legend_labels, legend_title="Action variability", legend_position= :bottomright) for i in eachindex(perceptualNoise)]


savefig.(noCommitBelief_pl, ["../figures/noCommitBelief_noise$i.png" for i in eachindex(perceptualNoise)])

## 1b) Perceptual noise vs action variability: performance
noCommit = selectdim(firstpass, 2, 2)
noCommitFP_pl = [groupedbar(noCommit[:,2:end,i], xlims=(10, nSteps+2), xticks=(vcat(10:10:nSteps-5, nSteps+1), vcat(10:10:nSteps-5, "Never")), ylims=(0,0.6),title="Noise = $(perceptualNoise[i])", xlabel="First pass (# steps)", ylabel="Proportion", label= reshape(legend_labels[2:end],1,:), linewidth=0, palette=cgrad(:acton, 5, rev=true, categorical=true)[2:5], legend_title="Action variability") for i in eachindex(perceptualNoise)]

savefig.(noCommitFP_pl, ["../figures/noCommitFP_noise$i.png" for i in eachindex(perceptualNoise)])

## 2) Same with commitment
withCommit = selectdim(belief, 2, 1) 

withCommitBelief_pl = [nice_plot(withCommit[:,:,i];title="Noise = $(perceptualNoise[i])", xlabel="Steps", ylabel="Belief", ylims=(0,1),label= legend_labels, legend_title="Action variability", legend_position= :bottomright) for i in eachindex(perceptualNoise)]

savefig.(withCommitBelief_pl, ["../figures/withCommitBelief_noise$i.png" for i in eachindex(perceptualNoise)])

##
withCommit = selectdim(firstpass, 2, 1)

withCommitFP_pl = [groupedbar(withCommit[:,2:end,i], xlims=(10, nSteps+2), xticks=(vcat(10:10:nSteps-5, nSteps+1), vcat(10:10:nSteps-5, "Never")), ylims=(0,0.6),title="Noise = $(perceptualNoise[i])", xlabel="First pass (# steps)", ylabel="Proportion", label=reshape(legend_labels[2:end],1,:), linewidth=0, palette=cgrad(:acton, 5, rev=true, categorical=true)[2:5], legend_title="Action variability") for i in eachindex(perceptualNoise)]

savefig.(withCommitFP_pl, ["../figures/withCommitFP_noise$i.png" for i in eachindex(perceptualNoise)])

## 2c) Perceptual noise vs action variability: commitment
withCommit = selectdim(commited, 2, 1)
withCommitC_pl = [nice_plot(withCommit[:,:,i];title="Noise = $(perceptualNoise[i])", xlabel="Steps", ylabel="Prob. maintained commitment", ylims = (0,1), label= legend_labels, legend_title="Action variability", legend_position= :topright) for i in eachindex(perceptualNoise)]

savefig.(withCommitC_pl, ["../figures/withCommitMC_noise$i.png" for i in eachindex(perceptualNoise)])

## 3) Behavioral switches 
CS = choiceSwitch[2:end,:,:,:] # Remove first trial
MCS = squeeze_mean(CS, dims=1)
choiceSwitches_pl = [groupedbar(MCS[:,:,i], palette = cgrad(:acton, 5, rev=true, categorical=true)[2:5], label=legend_labels, xticks = ([1,2], ["Commitment", "No Commitment"]), ylabel="Proportion of action switches", legend_title="Action variability", title="Noise = $(perceptualNoise[i])") for i in eachindex(perceptualNoise)]

savefig.(choiceSwitches_pl, ["../figures/choiceSwitches_noise$i.png" for i in eachindex(perceptualNoise)])