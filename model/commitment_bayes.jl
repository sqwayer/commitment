include("observer.jl")
include("utils.jl")

function leaky_update(logitp, ν)
    return logit(logistic(logitp) * (1-ν) + ν / 2)
end

function leaky_noisy_update(logitp, ν, ξ)
    return leaky_update(logitp, ν) + ξ*randn()
end

function p_commit(U, τ, θ)  
    return U^θ / (U^θ + τ^θ * (1 - U)^θ)
end

function action_selection(a1, a2, β)
    return logit(1 - cdf(Beta(β * a1, β * a2), 0.5))
end

""" Full model """

function full_model!(latent, obs; μ₀, μ₁, σ², λ, ξ)

    # Unpack latent variables
    Δ = latent[:Δ]
    a1 = latent[:a1]
    a2 = latent[:a2]
    resp = latent[:resp]
    lpfun = latent[:lpfun]

    # Log-likelihood ratio 
    llr = lpfun(μ₀, σ²[1], obs) - lpfun(μ₁, σ²[2], obs)

    # Belief updating 
    newresp, a1, a2 = observer_update(a1, a2, llr + ξ * randn(), λ)

    # Variability updating  
    #Δ *= 1 - α
    #Δ += α * dkl_from_logit(newresp, resp) / log(2)#(logistic(newresp) - logistic(resp))^2  #

    # Update latent variables
    latent[:Δ] = Δ
    latent[:a1] = a1
    latent[:a2] = a2
    latent[:resp] = newresp
    latent[:conf] = a1 + a2
    latent[:U] = 2 / (a1 + a2) # Normalized model uncertainty
    latent[:var] = a1 * a2 / (a1 + a2)^2 / (a1 + a2 + 1)
    latent[:λ] = λ

    return latent

end

function initialize_full_model!(latent, a0)

    latent[:Δ] = 0.0
    latent[:a1] = a0/2
    latent[:a2] = a0/2
    latent[:ai] = 0.0
    latent[:p̂] = 0.5
    latent[:resp] = 0.0
    latent[:conf] = a0
    latent[:U] = 2 / a0
    latent[:Uf] = 2 / a0
    latent[:var] = 1 / (4 * (a0+1))
    latent[:λ] = 0.0
    latent[:selectLP] = 0.0
    latent[:commited] = false
    latent[:commitTo] = 0

end

function initialize_full_model(a0)
    latent = Dict(
        :Δ => 0.0,
        :a1 => a0/2,
        :a2 => a0/2,
        :ai => 0.0,
        :p̂ => 0.5,
        :resp => 0.0,
        :conf => a0,
        :U => 2 / a0,
        :Uf => 2/a0,
        :var => 1 / (4 * (a0+1)),
        :λ => 0.0, 
        :selectLP => 0.0,
        :commited => false,
        :commitTo => 0,
        :lpfun => normal_llh
        )

    return latent

end

""" Partial model """
function partial_model!(latent, obs; μ, ρ, σ², λ, ξ)
    # Unpack latent variables
    ai = latent[:ai]
    lpfun = latent[:lpfun]

    # Log-likelihood ratio 
    llr = lpfun(μ, σ², obs) - log(ρ)

    # Belief updating 
    _, ai, _ = observer_update(ai, 1, llr + ξ * randn(), λ)

    # Update latent variables
    latent[:ai] = ai


    return latent
end

""" Update commitment and action selection """

function action_model!(latent, mdl; β, τ, θ)
    currentcommit = latent[:commited]
    if mdl == :full 
            latent[:selectLP] = logit(1 - cdf(Beta(β * latent[:a1], β * latent[:a2]), 0.5)) #β * log(latent[:a1]) - log(latent[:a2])
            latent[:commited] = false
            latent[:p̂] = latent[:a1] / (latent[:a1] + latent[:a2]) 
            latent[:Uf] = 2 / (latent[:a1] + latent[:a2]) # Foreground model uncertainty
    elseif mdl == :fixed
        fixed_belief_model_action!(latent, β, τ, θ)
    elseif mdl == :partial 
        partial_belief_model_action!(latent, β, τ, θ)
    end
    return latent[:commited] && currentcommit
end

""" Fixed belief model """

function fixed_belief_model_action!(latent, β, τ, θ)
    U = latent[:U]
    pC = p_commit(U, τ, θ)

    latent[:commited] = rand() < pC # Commit
    if latent[:commited]
        sgn = sign(latent[:selectLP])
        if sgn == 0.0
            latent[:selectLP] = rand((Inf, -Inf))
        else
            latent[:selectLP] = Inf * sgn
        end
        latent[:p̂] = logistic(latent[:selectLP])
        latent[:Uf] = 0.0

    else
        latent[:selectLP] = logit(1 - cdf(Beta(β * latent[:a1], β * latent[:a2]), 0.5))
        latent[:Uf] = 2 / (latent[:a1] + latent[:a2])
    end
end


""" Partial belief model """

function partial_belief_model_action!(latent, β, τ, θ)
    U = latent[:U]
    pC = p_commit(U, τ, θ)
    alreadyCommited = latent[:commited]

    latent[:commited] = rand() < pC # Commit

    if latent[:commited] 
        if !alreadyCommited # New commitment
            sgn = sign(latent[:selectLP]) # Direction of commitment
            if sgn > 0 || (sgn == 0 && rand() <= 0.5)
                # Commit to cat. 1
                latent[:ai] = latent[:a1]#0.5*(latent[:a1] + latent[:a2])
                latent[:commitTo] = 1

            else
                # Commit to cat. 2
                latent[:ai] = latent[:a2]#0.5*(latent[:a2] + latent[:a1])
                latent[:commitTo] = 2
                
            end

        # else # Already commited 
        #     pChg = p_commit(latent[:Uf], τ, θ) # Check uncertainty of the partial observer
        #     if rand() < pChg 
        #         if latent[:commitTo] == 1
        #             # Commit to cat. 2
        #             latent[:ai] = latent[:a2]
        #             latent[:commitTo] = 2

        #         else
        #             # Commit to cat. 1
        #             latent[:ai] = latent[:a1]
        #             latent[:commitTo] = 1

        #         end
        #     end
        end

        if latent[:commitTo] == 1
            # Commit to cat. 1
            latent[:selectLP] = action_selection(latent[:ai], 1, β)
            latent[:p̂] = latent[:ai] / (1 + latent[:ai])

        elseif latent[:commitTo] == 2
            # Commit to cat. 2
            latent[:selectLP] = action_selection(1, latent[:ai], β)
            latent[:p̂] = 1 / (1 + latent[:ai])
        end

        
        latent[:Uf] = 1 / (1 + latent[:ai])

    else
        latent[:selectLP] = action_selection(latent[:a1], latent[:a2], β)
        latent[:Uf] = 2 / (latent[:a1] + latent[:a2])
        latent[:p̂] = latent[:a1] / (latent[:a1] + latent[:a2]) 
    end

end
