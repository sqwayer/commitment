function observer_update(a1, a2, l, λ)

     # Mean belief from pseudocounts
    μ = log(a1) - log(a2) + l

    # Update pseudocounts
    a1 += logistic(μ)
    a2 += logistic(-μ)

    # Leak 
    a1 *= (1-λ)
    a1 += λ
    a2 *= (1-λ)
    a2 += λ

    return μ, a1, a2
end

function observer_commit(commited, sbelief; a1, a2, θ, β)
    H = 1/(a1+a2)
    belief = log(a1) - log(a2)

    currentcommit = false
    if H > θ 
        if !commited
            sbelief = belief == 0.0 ? rand((-Inf, Inf)) : Inf * sign(belief)
            commited = true
        else
            currentcommit = true
        end
    else
        sbelief = β * belief
        commited = false
    end

    return commited, currentcommit, sbelief
end

function multi_hyp_observer!(a, l, λ; ρ=0.0)
    
    # Mean belief from pseudocounts
    s = ρ
    r = zero(a)
    for i in eachindex(a) 
        r[i] = log(a[i]) + l[i] 
        s += exp(r[i])
    end
    r .-= log(s)

    # Update pseudocounts
    a .+= exp.(r) 

    # Leak 
    a .*= 1 - λ
    a .+= λ

    return r
end