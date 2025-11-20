include("utils.jl")

function forest_simple(nSteps, nIter; perceptualNoise = range(1.0, 5.0, length=10), inverseTemp = vcat(0.0, logrange(1.0, 50.0, length=9)), commitmentThresh = vcat(0.0, logrange(0.01, log(2), length=9)), Δ = 0.1, σ² = 1.0, ν = 0.0, ξ = 0.0)

    slen = length(perceptualVar)
    blen = length(inverseTemp)
    clen = length(commitmentThresh)
    
    # Pre-allocate
    belief = zeros(nSteps+1, clen, blen, slen)
    firstpass = zeros(nSteps+1, clen, blen, slen)
    commited = zeros(nSteps+1, clen, blen, slen)
    choiceSwitch = zeros(nSteps, clen, blen, slen)
    beliefAvg = zeros(nSteps+1, nIter)
    firstpassAvg = zeros(nSteps+1, nIter)
    commitAvg = zeros(nSteps+1, nIter)
    choiceAvg = zeros(nSteps, nIter)

    # Loop
    wb = Progress(slen*blen*clen, "Simulations....")
    for i = 1:slen 
        for j = 1:blen 
            for k = 1:clen 

                forest_simple_avg!(view(belief,:,k,j,i), view(firstpass,:,k,j,i), view(commited,:,k,j,i),view(choiceSwitch,:,k,j,i), beliefAvg, firstpassAvg, commitAvg, choiceAvg; Δ = Δ, σ² = σ², ν = ν, s = perceptualNoise[i], ξ = ξ, β = inverseTemp[j], θ = commitmentThresh[k]) 

                next!(wb)
            end
        end
    end

    return belief, firstpass, commited, choiceSwitch
end

function forest_simple_avg!(bvec, fvec, cvec, chvec, beliefAvg, firstpassAvg, commitAvg, choiceSwitch; Δ, σ², s, β, θ, ν, ξ)

    beliefAvg .= 0.0
    firstpassAvg .= 0.0
    commitAvg .= 0
    choiceSwitch .= 0

    for i in axes(beliefAvg, 2)
        forest_simple_run!(view(beliefAvg, :, i), view(firstpassAvg, :, i), view(commitAvg, :, i), view(choiceSwitch, :, i); Δ, σ², s, β, θ, ν, ξ)
    end

    bvec .= mean(beliefAvg, dims=2)
    fvec .= mean(firstpassAvg, dims=2)
    cvec .= mean(commitAvg, dims=2)
    chvec .= mean(choiceSwitch, dims=2)
end

function forest_simple_run!(belief, firstpass, currentcommit, choiceSwitch; Δ, σ², s, β, θ, ν, ξ)
    nSteps = length(belief) - 1

    x = 0.0
    xalt = 0.0
    selectLP = 0.0
    infvec = [-Inf, Inf]
    commited = false
    neverpassed = true
    lastchoice = -1

    wₛ = 1.0
    for t = 1:nSteps
        # Choice
        choice = rand(BernoulliLogit(selectLP))
        choiceSwitch[t] = choice ≠ lastchoice
        lastchoice = choice 

        # Update the generative process and observe outcome
        x += choice ? Δ : -Δ 
        xalt += !choice ? Δ : -Δ 

        μ₀ = tanh(x)
        μ₁ = tanh(xalt)
        #x = clamp(x, -1.0, 1.0)
        #xalt = clamp(xalt, -1.0, 1.0)
        obs = rand(Normal(μ₀, s))

        # Update agent's perceptual variance
        wₛ += 1
        σ² *= (1 - 1/wₛ)
        σ² += logistic(belief[t])/wₛ * (obs - μ₀)^2
        σ² += logistic(-belief[t])/wₛ * (obs - μ₁)^2

        # Update agent's model
        llr = logpdf(Normal(μ₀, sqrt(σ²)), obs) - logpdf(Normal(μ₁, sqrt(σ²)), obs)
        belief[t+1] = leaky_noisy_update(belief[t] + llr, ν, ξ)

        # Commitment
        H = entropy_from_logit(belief[t+1])
        currentcommit[t+1] = false
        if H > θ 
            if !commited
                selectLP = rand(infvec)
                commited = true
            else
                currentcommit[t+1] = true
            end
        else
            selectLP = β * belief[t+1]
            commited = false
        end

        belief[t] = logistic(belief[t]) # Convert to probability
        if neverpassed && x >= 1.0
            firstpass[t] = 1.0
            neverpassed = false
        end
    end


    belief[end] = logistic(belief[end])
    if neverpassed
        firstpass[end] = 1.0
    end

end