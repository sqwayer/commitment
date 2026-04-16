using ColorSchemes, Measures
# Color Palette 
PTols = ColorScheme([
    colorant"#228833",
    colorant"#AA3377",
    colorant"#4477AA",
    colorant"#CCBB44",
    colorant"#EE6677",
    colorant"#66CCEE",
    ]
)

PTolsGrad1(n) = ColorScheme(range(PTols[1], PTols[4], length=n))
PTolsGrad2(n) = ColorScheme(range(PTols[3], PTols[2], length=n))

nice_plot(X; palfn = PTolsGrad1, kw...) = begin
    pal = palfn(size(X, 2))
    plot(X; linewidth=3, palette = pal, ylims=(0.5,1), label="", size=(500, 500), dpi = 300, labelfontsize=14, tickfontsize=14, background_color=:transparent, kw...)
end

nice_bar_plot(X::Matrix; colpal = PTolsGrad1, kw...) = begin
    nSteps = size(X,1)
    pal = colpal(size(X, 2)) 

    groupedbar(X; xlims=(10, nSteps+2), xticks=(vcat(10:10:nSteps-5, nSteps+1), vcat(10:10:nSteps-5, "Never")), ylims=(0,0.8), xlabel="First pass (# steps)", ylabel="Proportion", linewidth=0, bar_width=1.4, palette=pal, label="", size=(600, 500), dpi = 300, labelfontsize=14, tickfontsize=14, background_color=:transparent, kw...)
end

nice_bar_plot(X::Vector; color = PTols[1], kw...) = begin
    nSteps = size(X,1)

    bar(X; xlims=(10, nSteps+2), xticks=(vcat(10:10:nSteps-5, nSteps+1), vcat(10:10:nSteps-5, "> $(nSteps-1)")), ylims=(0,0.8), xlabel="First pass (# steps)", ylabel="Proportion", linewidth=0, bar_width=1.4, color=color, label="", size=(600, 500), dpi = 300, labelfontsize=14, tickfontsize=14, background_color=:transparent, kw...)
end

quantile_plot!(X, Y, nbins; errbar = :sem, kw...) = begin
    xb = quantile_bins(X, nbins)
    xm = (xb[1:end-1] .+ xb[2:end]) ./ 2
    ym = zero(xm)
    ys = zero(xm)
    for i in 1:length(xb)-1
        idx = findall(xb[i] .<= X .< xb[i+1])
        ym[i] = mean(Y[idx])
        ys[i] = std(Y[idx])
        if errbar == :sem 
            ys[i] /= sqrt(length(idx))
        end
    end
    plot!(xm, ym; ribbon = ys, kw...)
end

quantile_plot(X, Y, nbins; errbar = :sem, kw...) = begin
    plot()
    quantile_plot!(X, Y, nbins; errbar = errbar, kw...)
end

nice_heatmap(X, Y, xbins::Int, ybins::Int; kw...) = begin
    xb = range(minimum(X), maximum(X), length=xbins)
    yb = range(minimum(Y), maximum(Y), length=ybins)
    nice_heatmap(X, Y, xb, yb; kw...)
end

nice_heatmap(X, Y, xb::T, yb::T; kw...) where T <:AbstractArray = begin
    M = zeros(length(xb), length(yb))
    for j in 1:length(yb)-1, i in 1:length(xb)-1
        M[i,j] = mean( (xb[i] .<= X .< xb[i+1]) .* (yb[j] .<= Y .< yb[j+1]) ) 
    end
    heatmap(xb, yb, M'; size=(500, 500), dpi = 300, labelfontsize=14, tickfontsize=14, background_color=:transparent, kw...)
end
