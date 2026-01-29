using SpecialFunctions


function ambiguous_obs(a1, a2, l, λ)
    # Mean belief from pseudocounts
    #p̂ = a1 / (a1 + a2)
    μ = log(a1) - log(a2) + l#logit(p̂) + l


    # Update pseudocounts
    a1 += logistic(μ)
    a2 += logistic(-μ)

    # Leak 
    a1 *= (1-λ)
    #a1 = max(a1, 0.1)
    a1 += λ
    a2 *= (1-λ)
    #a2 = max(a2, 0.1)
    a2 += λ

    return μ, a1, a2
end


function pseudocounts(μ, σ)

    p̂ = approx_mean_p(μ, σ)
    varp = approx_var_p(μ, σ)
    A = p̂ * (1 - p̂) / varp - 1
    
    return A, p̂, 1 - p̂
end

function approx_mean_p(μ, σ)
    return quasi_numerical_moment(1, μ, σ; K = 1000)
end

function approx_var_p(μ, σ)
    return quasi_numerical_moment(2, μ, σ; K = 1000) - quasi_numerical_moment(1, μ, σ; K = 1000) ^2
end

function test_sampling(niter = 1000, samples = 10_000)
    empiric = zeros(niter, 2)
    theoric = zeros(niter, 2)
    for i = 1:niter
        # α = rand(Gamma())
        # β = rand(Gamma())
        # B = Beta(α, β)
        # ν = trigamma(α) + trigamma(β)
        # L = rand(Normal())

        # lp = logit.(rand(B, samples))
        # lp_ = lp .+ L

        # μ = mean(lp)#digamma(α) - digamma(β)
        # α_ = α * logistic(L)
        # β_ = β + α * logistic(-L)
        # #B_ = Beta(α_, β_)

        μ = rand(Normal())
        σ = rand(LogNormal())

        lp = rand(Normal(μ, σ), samples)

        empiric[i, 1] = mean(logistic.(lp))
        empiric[i, 2] = var(logistic.(lp))
        theoric[i, 1] = logistic(μ)
        theoric[i, 2] = quasi_numerical_moment(2, μ, σ; K = 35) - quasi_numerical_moment(1, μ, σ; K = 35) ^2
    end
    return empiric, theoric
end

function quasi_numerical_moment(n, μ, σ; K = 10)
    est = 0.0
    D = Normal(μ, σ)
    for i in 1:K - 1
        est += logistic(quantile(D, i/K))^n
    end
    return est / (K-1)
end