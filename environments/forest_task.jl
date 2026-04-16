include("../model/commitment_bayes.jl")

function forest_task_simu(nSteps, nIter; darkZoneRadius = 0.5, stepsize = 0.1, σ² = 1.0, s = 1.0, ξ = 0.0, a0 = 2, λ = 0.0, θ = 1.0, τ = 1.0, β = 1.0, ρ = 1/2/pi, mdl = [:full])

    len = length(darkZoneRadius)
    mlen = length(mdl)
    
    # Pre-allocate
    beliefAvg = zeros(nSteps+1, len, mlen)
    firstpassAvg = zeros(nSteps+1, len, mlen)
    staycommitAvg = zeros(nSteps+1, len, mlen)
    confidenceAvg = zeros(nSteps+1, len, mlen)

    belief = zeros(nSteps+1, nIter)
    firstpass = zeros(nSteps+1, nIter)
    staycommit = zeros(nSteps+1, nIter)
    confidence = zeros(nSteps+1, nIter)

    # Loop
    wb = Progress(len * mlen, "Simulations....")
    for i = 1:len 
        for j = 1:mlen

            forest_task_avg!(view(beliefAvg,:,i, j), view(firstpassAvg,:,i, j), view(staycommitAvg,:,i, j),view(confidenceAvg,:,i, j), belief, firstpass, staycommit, confidence; darkZoneRadius = darkZoneRadius[i], stepsize = stepsize, a0 = a0, mdl = mdl[j], σ² = σ², s = s, ξ = ξ, β = β, τ = τ, θ = θ, λ = λ, ρ = ρ) 

        next!(wb)

        end
    end

    return beliefAvg, firstpassAvg, staycommitAvg, confidenceAvg
end

function forest_task_avg!(bvec, fvec, scvec, cvec, belief, firstpass, staycommit, confidence; darkZoneRadius, stepsize, a0, mdl, params...)

    belief .= 0.5
    firstpass .= 0.0
    staycommit .= 0
    confidence .= 2/a0

    Threads.@threads for i in axes(belief, 2)
        latent = initialize_full_model(a0)
        forest_task!(view(belief, :, i), view(firstpass, :, i), view(staycommit, :, i), view(confidence, :, i), latent, missing; mdl = mdl, darkZoneRadius = darkZoneRadius, stepsize = stepsize, params...)
    end

    bvec .= mean(belief, dims=2)
    fvec .= mean(firstpass, dims=2)
    scvec .= mean(staycommit, dims=2)
    cvec .= mean(confidence, dims=2)
end

function forest_task!(belief, firstpass, staycommit, confidence, latent, trace; mdl = :full, stepsize, darkZoneRadius, σ², s, τ, β, θ, ξ, λ, ρ)

    nSteps = length(belief) - 1

    x = 0.0
    xalt = 0.0
    neverpassed = true

    if isa(σ², Real)
        σ² = fill(σ², 2)
    end

    for t = 1:nSteps   
        if !ismissing(trace)
            trace[t] = x 
        end

        # Choice
        choice = rand(BernoulliLogit(latent[:selectLP]))

        # Update the generative process and observe outcome
        x += choice ? stepsize : -stepsize
        xalt += !choice ? stepsize : -stepsize

        μ₀ = abs(x) >= darkZoneRadius ? clamp(x, -1, 1) : 0.0#tanh(x)#
        μ₁ = abs(xalt) >= darkZoneRadius ? clamp(xalt, -1, 1) : 0.0#tanh(xalt)#
        obs = rand(Normal(μ₀, s))

        # Update full model
        full_model!(latent, obs; μ₀ = μ₀, μ₁ = μ₁, σ² = σ², λ = λ, ξ = ξ)

        # Update foreground model 
        if mdl == :partial && latent[:commited]
            sgn = sign(latent[:selectLP]) # Get the commited belief
            if sgn > 0.0
                partial_model!(latent, obs; μ = μ₀, ρ = ρ, σ² = σ²[1], λ = λ, ξ = ξ)
            elseif sgn < 0.0
                partial_model!(latent, obs; μ = μ₁, ρ = ρ, σ² = σ²[2], λ = λ, ξ = ξ)
            else
                error("sgn = 0")
            end
        end

        # Commit/Action selection
        staycommit[t+1] = action_model!(latent, mdl; β = β, τ = τ, θ = θ)

        # Track variables 
        belief[t+1] = latent[:p̂]
        confidence[t+1] = latent[:Uf]

        # Check first pass
        if neverpassed && x >= 1.0
            firstpass[t] = 1.0
            neverpassed = false
        end
    end

    if neverpassed
        firstpass[end] = 1.0
    end

end