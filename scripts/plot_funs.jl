nice_plot(X; kw...) = plot(X; linewidth=3, palette = cgrad(:algae, size(X, 2)+1, rev=false, categorical=true)[2:size(X,2)+1], ylims=(0.5,1), label="", size=(500, 500), dpi = 300, background_color=:transparent, kw...)

nice_bar_plot(X; kw...) = begin
    nSteps = size(X,1)
    groupedbar(X; xlims=(10, nSteps+2), xticks=(vcat(10:10:nSteps-5, nSteps+1), vcat(10:10:nSteps-5, "Never")), ylims=(0,0.8), xlabel="First pass (# steps)", ylabel="Proportion", linewidth=0, palette=cgrad(:algae, 5, rev=false, categorical=true)[2:5], label="", size=(500, 500), dpi = 300,background_color=:transparent, kw...)
end


