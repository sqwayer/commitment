include("../model/commitment_bayes.jl")


function seq_task_simu(nSteps, nIter; epLength = nSteps, lesionInt = [0], process = VonMises, σ² = 1.0, s = 1.0, ξ = 0.0, a0 = 2, λ = 0.0, λ₂ = 0.0, θ = 1.0, τ = 1.0, β = 1.0, ρ = 1/2/pi, κ = 1.0, mdl = [:full])

    mlen = length(mdl)
    
    # Pre-allocate
    beliefAvg = zeros(nSteps+1, mlen)
    firstpassAvg = zeros(nSteps+1, mlen)
    staycommitAvg = zeros(nSteps+1, mlen)
    confidenceAvg = zeros(nSteps+1, mlen)

    belief = zeros(nSteps+1, nIter)
    firstpass = zeros(nSteps+1, nIter)
    staycommit = zeros(nSteps+1, nIter)
    confidence = zeros(nSteps+1, nIter)

    # Loop
    wb = Progress(mlen, "Simulations....")
    for i = 1:mlen 

        seq_task_avg!(view(beliefAvg,:,i), view(firstpassAvg,:,i), view(staycommitAvg,:,i),view(confidenceAvg,:,i), belief, firstpass, staycommit, confidence; epLength = epLength, lesionInt = lesionInt, process = process, a0 = a0, mdl = mdl[i], σ² = σ², s = s, ξ = ξ, β = β, τ = τ, θ = θ, λ = λ, λ₂ = λ₂, ρ = ρ, κ = κ) 

        next!(wb)

    end

    return beliefAvg, firstpassAvg, staycommitAvg, confidenceAvg
end

function seq_task_avg!(bvec, fvec, scvec, cvec, belief, firstpass, staycommit, confidence; epLength, lesionInt, process, a0, mdl, params...)
    
    belief .= 0.5
    firstpass .= 0.0
    staycommit .= 0
    confidence .= 2/a0

    Threads.@threads for i in axes(belief, 2)
        latent = initialize_full_model(a0)
        if process == Normal
            latent[:lpfun] = normal_llh
        elseif process == VonMises
            latent[:lpfun] = vonmises_llh
        elseif process == Bernoulli
            latent[:lpfun] = bernoulli_llh
        end
    
        seq_task!(view(belief, :, i), view(firstpass, :, i), view(staycommit, :, i), view(confidence, :, i), latent; mdl = mdl, epLength = epLength, lesionInt = lesionInt, process = process, params...)
    end

    bvec .= mean(belief, dims=2)
    fvec .= mean(firstpass, dims=2)
    scvec .= mean(staycommit, dims=2)
    cvec .= mean(confidence, dims=2)

end

function seq_task!(belief, firstpass, staycommit, confidence, latent; mdl = :full, epLength, lesionInt, process, σ², s, τ, β, θ, ξ, λ, λ₂, ρ, κ)

    nSteps = length(belief) - 1
    neverpassed = true

    μ = pi/2
    μ₀ = pi/2
    μ₁ = -pi/2  
    cat = 1
    if isa(s, Real)
        s = fill(s, 2)
    end

    if isa(σ², Real)
        σ² = fill(σ², 2)
    end

    for t = 1:nSteps

        obs = process == Bernoulli ? rand(BernoulliLogit(s[1] * μ)) : rand(process(μ, s[cat]))

        # Check leak 
        lam = in(t, lesionInt) ? λ₂ : λ

        # Update full model
        staycommit[t] = latent[:commited]# ? (-1)^latent[:commitTo] : 0.0
        full_model!(latent, obs; μ₀ = μ₀, μ₁ = μ₁, σ² = σ², λ = lam, ξ = ξ)

        # Update foreground model 
        if mdl == :partial && latent[:commited]
            if latent[:commitTo] == 1
                partial_model!(latent, obs; μ = μ₀, ρ = ρ, σ² = σ²[1], λ = lam, ξ = ξ)
            elseif latent[:commitTo] == 2
                partial_model!(latent, obs; μ = μ₁, ρ = ρ, σ² = σ²[2], λ = lam, ξ = ξ)
            else
                error("commit not set")
            end
        end

        # Commit ?/Action probability ? 
        action_model!(latent, mdl; β = β, τ = τ, θ = θ)

        # Track variables 
        belief[t+1] = latent[:p̂]
        confidence[t+1] = latent[:Uf]

        # Reversal
        if mod(t, epLength) == 0
            μ *= -1
            cat = mod(cat, 2)+1
        end

        # Check first pass (decision to stop in beads task)
        amax = max(latent[:a1], latent[:a2])
        amin = min(latent[:a1], latent[:a2])
        stopP = 1 - cdf(Beta(amax, amin), κ)
        if mdl == :partial && latent[:commited]
            stopP = 1 - cdf(Beta(latent[:ai], 1), κ)
        elseif mdl == :fixed && latent[:commited]
            stopP = 1.0
        end
        # if mdl == :partial 
        #     abslog = abs(logit(cdf(Beta(1 + β * latent[:ai], 1 + β), 0.5)))
        # else
        #     abslog = abs(logit(cdf(Beta(1 + β * latent[:a1], 1 + β * latent[:a2]), 0.5)))
        # end
        #stopP = (logistic(abs(latent[:selectLP])) - 0.5) * 2#1 - exp(- abslog)
        #
        if neverpassed && rand(Bernoulli(stopP))
            firstpass[t] = 1.0
            neverpassed = false
        end

    end

    if neverpassed
        firstpass[end] = 1.0
    end

end