using StatsFuns

entropy_from_logit(L) = begin
    p = logistic(L)
    return -(L * (p - 1) + log(p))
end

squeeze_mean(X; dims) = dropdims(mean(X, dims=dims), dims=dims)

nice_plot(X; kw...) = plot(X; linewidth=3, palette = cgrad(:acton, size(X, 2)+1, rev=true, categorical=true)[2:size(X,2)+1], ylims=(0.5,1), label="", size=(500, 500), dpi = 300, kw...)


