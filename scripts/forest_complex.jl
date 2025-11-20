
function forest_complex(nSteps, nStates, nIter; perceptualVar = range(1.0, 5.0, length=10), inverseTemp = vcat(0.0, logrange(1.0, 50.0, length=9)), commitmentThresh = vcat(0.0, logrange(0.01, log(2), length=9)), Δ = 0.1, σ² = 1.0)

    slen = length(perceptualVar)
    blen = length(inverseTemp)
    clen = length(commitmentThresh)
    
    # Pre-allocate
    stateSz = fill(2, nStates)
    belief = zeros(nSteps+1, stateSz..., clen, blen, slen)
    firstpass = zeros(nSteps+1, stateSz..., clen, blen, slen)
    beliefAvg = zeros(nSteps+1, stateSz..., nIter)
    firstpassAvg = zeros(nSteps+1, stateSz..., nIter)

    # Loop
    wb = Progress(slen*blen*clen, "Simulations....")

    for (bel, fp) in zip(eachslice(belief, dims=(nStates+2,nStates+3,nStates+4)), eachslice(firstpass, dims=(nStates+2, nStates+3, nStates+4)))

        forest_complex_avg!(bel, fp, beliefAvg, firstpassAvg; Δ = Δ, σ² = σ², s² = perceptualVar[1], β = inverseTemp[1], θ = commitmentThresh[1])

        next!(wb)
    end


    return belief, firstpass
end

function forest_complex_avg!(bvec, fvec, beliefAvg, firstpassAvg; Δ, σ², s², β, θ)

    dimSz = size(beliefAvg)
    iterDim = length(dimSz)
    nIter = dimSz[end]

    beliefAvg .= 0.0
    firstpassAvg .= 0.0

    for i in 1:nIter
        forest_complex_run!(selectdim(beliefAvg, iterDim, i), selectdim(firstpassAvg, iterDim, i); Δ, σ², s², β, θ)
    end


    bvec .= mean(beliefAvg, dims=iterDim)
    fvec .= mean(firstpassAvg, dims=iterDim)
end


function forest_complex_run!(belief, firstpass; Δ, σ², s², β, θ)
    nSteps = size(belief,1) - 1

    x = 0.0
    selectLP = 0.0
    infvec = [-Inf, Inf]
    commited = false
    neverpassed = true
    for t = 1:nSteps
        # Choice
        choice = rand(BernoulliLogit(selectLP))

        # Update the generative process and observe outcome
        x += choice ? Δ : -Δ 
        obs = rand(Normal(x, s²))

        # Update agent's model
        
        llr = logpdf(Normal(x, σ²), obs) - logpdf(Normal(-x, σ²), obs)
        belief[t+1,:] = belief[t,:] .+ llr

        # Commitment
        H = entropy_from_logit(belief[t+1,1])
        if H > θ 
            if !commited
                selectLP = rand(infvec)
                commited = true
            end
        else
            selectLP = β * belief[t+1,1]
            commited = false
        end

        belief[t,:] .= logistic.(belief[t,:]) # Convert to probability
        if neverpassed && x >= 1.0
            firstpass[t,:] .= 1.0
            neverpassed = false
        end
    end
    belief[end,:] = logistic.(belief[end,:])
    if neverpassed
        firstpass[end,:] .= 1.0
    end
end