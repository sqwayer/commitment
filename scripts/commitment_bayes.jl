function leaky_update(logitp, ν)
    return logit(logistic(logitp) * (1-ν) + ν / 2)
end

function leaky_noisy_update(logitp, ν, ξ)
    return leaky_update(logitp, ν) + ξ*randn()
end

function commitment!(logP, dim, sel)
    Z = 0.0
    for i in eachindex(logP)
        if CartesianIndices(logP)[i].I[dim] ≠ sel 
            logP[i] = -Inf
        else
            Z += exp(logP[i])
        end
    end
    logP .-= log(Z)
end

function commitment_rand!(logP, dim)
    dimcard = size(logP)
    sel = rand(1:dimcard[dim])

    commitment!(logP, dim, sel)
end

function commitment_max!(logP, dim)
    dimcard = size(logP)
    m = -Inf
    sel = 0
    for j in axes(logP, dim)
        sj = logsumexp(selectdim(logP, dim, j))
        if sj > m 
            m = sj
            sel = j 
        end
    end

    commitment!(logP, dim, sel)
end