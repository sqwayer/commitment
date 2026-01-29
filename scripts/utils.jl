using StatsFuns

entropy_from_logit(L) = begin
    p = logistic(L)
    return -(L * (p - 1) + log(p))
end

dkl_from_logit(Lp, Lq) = begin
    p = logistic(Lp)
    return p * (Lp - Lq) + Lq - Lp + log(1 + exp(-Lq)) - log(1 + exp(-Lp))
end

squeeze_mean(X; dims) = dropdims(mean(X, dims=dims), dims=dims)


""" Log-likelihood functions """
normal_llh(μ, σ², x) = logpdf(Normal(μ, σ²), x) 
vonMiseslogpdf(d, x) = (d.κ * (cos(x - d.μ) - 1)) - log(twoπ * d.I0κx)
vonmises_llh(μ, κ, x) = vonMiseslogpdf(VonMises(μ, κ), x)
bernoulli_llh(μ, κ, x) = logpdf(BernoulliLogit(κ * μ), x)

