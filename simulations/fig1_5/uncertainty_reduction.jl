using Distributions, StatsFuns, StatsPlots, ProgressMeter

include("../../model/observer.jl")
include("../../model/utils.jl")
include("../plot_funs.jl")

## Hypotheses space 
Nhyp = 5
μ = range(0, 2*pi, length=Nhyp+1)[1:end-1]

# True distribution and observations 
trueDist = VonMises(0.0, 1.0)
Nobs = 100

# Simulations parameters 
λ = 0.1
niter = 1000

A = [ones(n, Nobs) for n = 1:Nhyp]
U = zeros(Nobs, Nhyp, niter)
V = zeros(Nhyp, niter)
Kap = zeros(Nhyp, niter)
SS1 = falses(niter)

wb = Progress(niter)
for ni = 1:niter
    # Generate observations 
    obs = rand(trueDist, 1, Nobs)

    # Lkelihoods
    κ = Nhyp * rand(Nhyp)
    lhs = vonmises_llh.(μ, κ, obs)

    # Select subsets 
    subsets = [sample(1:Nhyp, n, replace=false) for n = 1:Nhyp]

    V[:,ni] = [entropy_mixture_vm(μ[ss], κ[ss]) for ss in subsets]
    Kap[:,ni] = κ
    SS1[ni] = subsets[1] == [1]

    for t = 1:Nobs
        Threads.@threads for n = 1:Nhyp
            if t == 1
                A[n] .= 1.0
            end

            ρ = n == Nhyp ? 0.0 : 1/2/pi

            multi_hyp_observer!(@views(A[n][:,t]), lhs[subsets[n],t], λ, ρ=ρ)

            if t < Nobs
                A[n][:,t+1] = A[n][:,t]
            end

            U[t,n,ni] = n / sum(A[n][:,t])
        end
    end
    next!(wb)
end

##
mU = dropdims(mean(U, dims=3), dims=3)
sU = dropdims(std(U, dims=3) ./ sqrt(niter), dims=3)
maxU = dropdims(maximum(U, dims=3), dims=3)
minU = dropdims(minimum(U, dims=3), dims=3)

xwin = (0, 20)
cpal = PTols[[4, 5, 2, 1, 3]]#palette(reverse(vcat(PTols[3], PTolsGrad1(4)...)))#palette(reverse(PTolsGrad1(Nhyp)))
labels = reshape([" $i" for i = 1:Nhyp], 1 ,:)
plot(mU, ribbon = sU, linewidth=5, palette=cpal, xlims=xwin, ylims=(0,1), xticks=[], yticks=(0.2:0.2:0.8), label=labels, xlabel="# Observations", ylabel="Uncertainty", legend_title = "# Considered hypotheses", legendfontsize=14, legendtitlefontsize=14, labelfontsize=14, ytickfontsize=14, grid=false, background_color=:transparent, size=(500, 500), dpi=300)


##
idx = eachindex(SS1)#findall(.!SS1)
stationnaryU = U[end,:,:]
cpal = PTols[[4, 5, 2, 1, 3]]#palette(reverse(PTolsGrad2(Nhyp)))#palette(reverse(vcat(PTols[3], PTolsGrad1(4)...)))#

scatter(.-V[1,idx], stationnaryU[1,idx], msw=0, ms=2, color=cpal[1], label="", alpha=0.6)
for i = 2:Nhyp
    scatter!(.-V[i,:], stationnaryU[i,:], msw=0, ms=2, color=cpal[i], label="", alpha=0.6)
end

quantile_plot!(.-V[1,idx], stationnaryU[1,idx], 10, color=cpal[1], linewidth=5, label=" 1")
for i = 2:Nhyp
    quantile_plot!(.-V[i,:], stationnaryU[i,:], 10, color=cpal[i], linewidth=5, label=" $i")
end
plot!(xlabel = "Mixture Entropy", ylabel = "Uncertainty", legend_title = "# Considered hypotheses", xticks=0.8:0.3:2.5, legendfontsize=14, legendtitlefontsize=14, labelfontsize=14, tickfontsize=14, legend_position = :topright, grid=false, background_color = :transparent, size=(600, 500), dpi=300)
#hline!([stationnaryU[end,1]]; ylims=(0.0,1), xlims=(0.1, 0.4), color=cpal[end], linewidth=5, label="$Nhyp/$Nhyp")