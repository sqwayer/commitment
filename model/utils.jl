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

quantile_bins(X, nbins) = quantile(X, range(0, 1, length=nbins))

""" Log-likelihood functions """
normal_llh(μ, σ², x) = logpdf(Normal(μ, σ²), x) 
vonMiseslogpdf(d, x) = (d.κ * (cos(x - d.μ) - 1)) - log(twoπ * d.I0κx)
vonmises_llh(μ, κ, x) = vonMiseslogpdf(VonMises(μ, κ), x)
bernoulli_llh(μ, κ, x) = logpdf(BernoulliLogit(κ * μ), x)

""" Numerical computation of the entropy of a mixture of Von Mises"""
function entropy_mixture_vm(μ, κ; iter=100_000)
    M = MixtureModel([VonMises(μ[i], κ[i]) for i in eachindex(μ)])
    H = 0.0
    for i = 1:iter 
        u = rand(M)
        H += logpdf(M, u)
    end
    return H / iter
end

