using Oceananigans
using Oceananigans.Units
using Printf
using JLD2
using Statistics
using CUDA
using CairoMakie
using BenchmarkTools
using Dates
using Oceananigans: prognostic_fields

function initial_conditions!(model)
    δx = model.grid.Δxᶜᵃᵃ
    U = 1
    U_bg = 0

    R = 0.1      # width/radius of the broad center
    δ = 3δx     # dropoff thickness; smaller = sharper

    z₀ = 0.65

    r(x, y, z) = sqrt((z - z₀)^2)

    uᵢ(x, y, z) = U_bg + (U - U_bg) * 0.5 * (1 - tanh((r(x, y, z) - R) / δ))

    set!(model, u=uᵢ)
end

function setup_grid(Nx, Ny, Nz)
    grid = RectilinearGrid(GPU(), Float64,
                           size = (Nx, Ny, Nz),
                           halo = (5, 5, 5),
                           x = (0, 1),
                           y = (0, 1),
                           z = (0, 1),
                           topology = (Bounded, Periodic, Bounded))
    return grid
end

function setup_model(grid, timestepper=:RungeKutta3)
    closure = ScalarDiffusivity(ν=1e-5, κ=1e-5)

    model = NonhydrostaticModel(grid;
                                advection = UpwindBiased(order=advection_order),
                                timestepper,
                                closure)

    initial_conditions!(model)
    return model
end

Nx = Ny = Nz = 256
Δx = 1 / Nx
advection_order = 9
filename_prefix = "shear_flow_upwindbiased$(advection_order)"

grid = setup_grid(Nx, Ny, Nz)

cfls = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
Δts = cfls .* Δx ./ 1.6
Δt_ref = Δts[end] / 8

stop_time = 10

# These names match the run script filenames, while plot labels still use FPJ0/1/2.
timesteppers = [:RungeKutta3, :ConstantPressureProjectionRungeKutta3, :LinearPressureProjectionRungeKutta3, :MidpointPressureProjectionRungeKutta3]

const DATA_DIR = joinpath(@__DIR__, "data")
const OUTPUT_DIR = joinpath(@__DIR__, "output")
mkpath(OUTPUT_DIR)
#%%
rk_filenames = ["$(filename_prefix)_RungeKutta3_Nx$(Nx)_dt$(Δt).jld2" for Δt in Δts]
l0_filenames = ["$(filename_prefix)_ConstantPressureProjectionRungeKutta3_Nx$(Nx)_dt$(Δt).jld2" for Δt in Δts]
l1_filenames = ["$(filename_prefix)_LinearPressureProjectionRungeKutta3_Nx$(Nx)_dt$(Δt).jld2" for Δt in Δts]
l2_filenames = ["$(filename_prefix)_MidpointPressureProjectionRungeKutta3_Nx$(Nx)_dt$(Δt).jld2" for Δt in Δts]
ref_filename = "$(filename_prefix)_RungeKutta3_Nx$(Nx)_dt$(Δt_ref).jld2"

rk_u_datas = [FieldTimeSeries(joinpath(DATA_DIR, f), "u") for f in rk_filenames]
rk_w_datas = [FieldTimeSeries(joinpath(DATA_DIR, f), "w") for f in rk_filenames]

l0_u_datas = [FieldTimeSeries(joinpath(DATA_DIR, f), "u") for f in l0_filenames]
l0_w_datas = [FieldTimeSeries(joinpath(DATA_DIR, f), "w") for f in l0_filenames]
l1_u_datas = [FieldTimeSeries(joinpath(DATA_DIR, f), "u") for f in l1_filenames]
l1_w_datas = [FieldTimeSeries(joinpath(DATA_DIR, f), "w") for f in l1_filenames]
l2_u_datas = [FieldTimeSeries(joinpath(DATA_DIR, f), "u") for f in l2_filenames]
l2_w_datas = [FieldTimeSeries(joinpath(DATA_DIR, f), "w") for f in l2_filenames]

ref_u_data = FieldTimeSeries(joinpath(DATA_DIR, ref_filename), "u")
ref_w_data = FieldTimeSeries(joinpath(DATA_DIR, ref_filename), "w")

# All convergence errors are measured against the fine RK3 reference.
u_ref = interior(ref_u_data[2], :, :, :)
w_ref = interior(ref_w_data[2], :, :, :)

function final_time_errors(ut, wt)
    # n = length(ut.times)
    n = 2
    try
        u_num = interior(ut[n], :, :, :)
        w_num = interior(wt[n], :, :, :)

        eu = u_num .- u_ref
        ew = w_num .- w_ref

        L2_u = sqrt(mean(eu.^2))
        L2_w = sqrt(mean(ew.^2))

        L∞_u = maximum(abs, eu)
        L∞_w = maximum(abs, ew)

        return (; L∞_u = maximum(abs, eu), L∞_w = maximum(abs, ew),
                L2_u = sqrt(mean(eu.^2)), L2_w = sqrt(mean(ew.^2)),
                L2 = sqrt(L2_u^2 + L2_w^2), L∞ = max(L∞_u, L∞_w))
    catch e
        @warn "Error computing final time errors: $(typeof(e))"
        return (; L∞_u = NaN, L∞_w = NaN, L2_u = NaN, L2_w = NaN, L2 = NaN, L∞ = NaN)
    end
end
#%%
rk_L2_errs = zeros(length(rk_u_datas))
rk_L∞_errs = zeros(length(rk_u_datas))

l0_L2_errs = zeros(length(l0_u_datas))
l0_L∞_errs = zeros(length(l0_u_datas))

l1_L2_errs = zeros(length(l1_u_datas))
l1_L∞_errs = zeros(length(l1_u_datas))

l2_L2_errs = zeros(length(l2_u_datas))
l2_L∞_errs = zeros(length(l2_u_datas))

rk_L2_u_errs = zeros(length(rk_u_datas))
rk_L2_w_errs = zeros(length(rk_w_datas))
rk_L∞_u_errs = zeros(length(rk_u_datas))
rk_L∞_w_errs = zeros(length(rk_w_datas))

l0_L2_u_errs = zeros(length(l0_u_datas))
l0_L2_w_errs = zeros(length(l0_w_datas))
l0_L∞_u_errs = zeros(length(l0_u_datas))
l0_L∞_w_errs = zeros(length(l0_w_datas))

l1_L2_u_errs = zeros(length(l1_u_datas))
l1_L2_w_errs = zeros(length(l1_w_datas))
l1_L∞_u_errs = zeros(length(l1_u_datas))
l1_L∞_w_errs = zeros(length(l1_w_datas))

l2_L2_u_errs = zeros(length(l2_u_datas))
l2_L2_w_errs = zeros(length(l2_w_datas))
l2_L∞_u_errs = zeros(length(l2_u_datas))
l2_L∞_w_errs = zeros(length(l2_w_datas))

for i in eachindex(rk_u_datas)
    rk_errs = final_time_errors(rk_u_datas[i], rk_w_datas[i])
    l0_errs = final_time_errors(l0_u_datas[i], l0_w_datas[i])
    l1_errs = final_time_errors(l1_u_datas[i], l1_w_datas[i])
    l2_errs = final_time_errors(l2_u_datas[i], l2_w_datas[i])

    rk_L2_u_errs[i] = rk_errs.L2_u
    rk_L2_w_errs[i] = rk_errs.L2_w
    rk_L∞_u_errs[i] = rk_errs.L∞_u
    rk_L∞_w_errs[i] = rk_errs.L∞_w
    l0_L2_u_errs[i] = l0_errs.L2_u
    l0_L2_w_errs[i] = l0_errs.L2_w
    l0_L∞_u_errs[i] = l0_errs.L∞_u
    l0_L∞_w_errs[i] = l0_errs.L∞_w
    l1_L2_u_errs[i] = l1_errs.L2_u
    l1_L2_w_errs[i] = l1_errs.L2_w
    l1_L∞_u_errs[i] = l1_errs.L∞_u
    l1_L∞_w_errs[i] = l1_errs.L∞_w
    l2_L2_u_errs[i] = l2_errs.L2_u
    l2_L2_w_errs[i] = l2_errs.L2_w
    l2_L∞_u_errs[i] = l2_errs.L∞_u
    l2_L∞_w_errs[i] = l2_errs.L∞_w

    rk_L2_errs[i] = rk_errs.L2
    rk_L∞_errs[i] = rk_errs.L∞

    l0_L2_errs[i] = l0_errs.L2
    l0_L∞_errs[i] = l0_errs.L∞

    l1_L2_errs[i] = l1_errs.L2
    l1_L∞_errs[i] = l1_errs.L∞

    l2_L2_errs[i] = l2_errs.L2
    l2_L∞_errs[i] = l2_errs.L∞
end

function fit_power_law(Δts, errors)
    valid = isfinite.(errors) .& (errors .> 0)

    if sum(valid) < 2
        return NaN, fill(NaN, length(errors))
    end

    log_Δts = log10.(Δts[valid])
    log_errors = log10.(errors[valid])

    slope = sum((log_Δts .- mean(log_Δts)) .* (log_errors .- mean(log_errors))) /
            sum((log_Δts .- mean(log_Δts)).^2)

    intercept = mean(log_errors) - slope * mean(log_Δts)
    fit_errors = 10 .^ (intercept .+ slope .* log10.(Δts))

    return slope, fit_errors
end
#%%
cfls_for_ticks = [0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05]
Δts_for_ticks = cfls_for_ticks .* Δx ./ 1.6
cfl_labels = String[string(round(c, digits=2)) for c in cfls_for_ticks]

fig = Figure(size=(900, 500), fontsize=15)
axL2 = Axis(fig[1, 1], xlabel="Δt", ylabel="L2 error", xscale=log10, yscale=log10, xticklabelcolor=:blue, xlabelcolor=:blue, xtickcolor=:blue)
axL∞ = Axis(fig[1, 2], xlabel="Δt", ylabel="L∞ error", xscale=log10, yscale=log10, xticklabelcolor=:blue, xlabelcolor=:blue, xtickcolor=:blue)
axL2_2 = Axis(fig[1, 1], xscale=log10, xticklabelcolor=:red, xlabel="Advective CFL", xaxisposition=:top, xlabelcolor=:red, xtickcolor=:red, xticks=(Δts_for_ticks, cfl_labels))
axL∞_2 = Axis(fig[1, 2], xscale=log10, xticklabelcolor=:red, xlabel="Advective CFL", xaxisposition=:top, xlabelcolor=:red, xtickcolor=:red, xticks=(Δts_for_ticks, cfl_labels))

linkxaxes!(axL2, axL2_2)
linkxaxes!(axL∞, axL∞_2)

hidespines!(axL2_2)
hidespines!(axL∞_2)
hideydecorations!(axL2_2)
hideydecorations!(axL∞_2)
hidexdecorations!(axL2_2, ticks=false, ticklabels=false, label=false)
hidexdecorations!(axL∞_2, ticks=false, ticklabels=false, label=false)

l0_fit_start = 5
l1_fit_start = 7
l2_fit_start = 7
rk_fit_start = 4
# Fit starts exclude unstable or saturated large-CFL points.
markersize = 13
linewidth = 2

l0_L2_slope, l0_L2_fit = fit_power_law(Δts[l0_fit_start:end], l0_L2_errs[l0_fit_start:end])
l0_L∞_slope, l0_L∞_fit = fit_power_law(Δts[l0_fit_start:end], l0_L∞_errs[l0_fit_start:end])
l1_L2_slope, l1_L2_fit = fit_power_law(Δts[l1_fit_start:end], l1_L2_errs[l1_fit_start:end])
l1_L∞_slope, l1_L∞_fit = fit_power_law(Δts[l1_fit_start:end], l1_L∞_errs[l1_fit_start:end])
l2_L2_slope, l2_L2_fit = fit_power_law(Δts[l2_fit_start:end], l2_L2_errs[l2_fit_start:end])
l2_L∞_slope, l2_L∞_fit = fit_power_law(Δts[l2_fit_start:end], l2_L∞_errs[l2_fit_start:end])
rk_L2_slope, rk_L2_fit = fit_power_law(Δts[rk_fit_start:end], rk_L2_errs[rk_fit_start:end])
rk_L∞_slope, rk_L∞_fit = fit_power_law(Δts[rk_fit_start:end], rk_L∞_errs[rk_fit_start:end])

scatter!(axL2, Δts, l0_L2_errs, markersize=markersize, marker=:circle)
scatter!(axL2, Δts, l1_L2_errs, markersize=markersize, marker=:rect)
scatter!(axL2, Δts, l2_L2_errs, markersize=markersize, marker=:utriangle)
scatter!(axL2, Δts, rk_L2_errs, markersize=markersize, marker=:cross)

lines!(axL2, Δts[l0_fit_start:end], l0_L2_fit, label="FPJ0 (slope = $(round(l0_L2_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL2, Δts[l1_fit_start:end], l1_L2_fit, label="FPJ1 (slope = $(round(l1_L2_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL2, Δts[l2_fit_start:end], l2_L2_fit, label="FPJ2 (slope = $(round(l2_L2_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL2, Δts[rk_fit_start:end], rk_L2_fit, label="RK3 (slope = $(round(rk_L2_slope, sigdigits=3)))", linewidth=linewidth, linestyle=:dash)

scatter!(axL∞, Δts, l0_L∞_errs, markersize=markersize, marker=:circle)
scatter!(axL∞, Δts, l1_L∞_errs, markersize=markersize, marker=:rect)
scatter!(axL∞, Δts, l2_L∞_errs, markersize=markersize, marker=:utriangle)
scatter!(axL∞, Δts, rk_L∞_errs, markersize=markersize, marker=:cross)

lines!(axL∞, Δts[l0_fit_start:end], l0_L∞_fit, label="FPJ0 (slope = $(round(l0_L∞_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL∞, Δts[l1_fit_start:end], l1_L∞_fit, label="FPJ1 (slope = $(round(l1_L∞_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL∞, Δts[l2_fit_start:end], l2_L∞_fit, label="FPJ2 (slope = $(round(l2_L∞_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL∞, Δts[rk_fit_start:end], rk_L∞_fit, label="RK3 (slope = $(round(rk_L∞_slope, sigdigits=3)))", linewidth=linewidth, linestyle=:dash)

axislegend(axL2, position=:rb)
axislegend(axL∞, position=:rb)
display(fig)
# save(joinpath(OUTPUT_DIR, "$(filename_prefix)_combined_error_convergence.png"), fig, px_per_unit=4)
# save(joinpath(OUTPUT_DIR, "$(filename_prefix)_combined_error_convergence.pdf"), fig)
#%%
fig = Figure(size=(900, 500), fontsize=15)
axL2 = Axis(fig[1, 1], title="L2 error at final time", xlabel="Δt", ylabel="L2 error", xscale=log10, yscale=log10)
axL∞ = Axis(fig[1, 2], title="L∞ error at final time", xlabel="Δt", ylabel="L∞ error", xscale=log10, yscale=log10)

l0_fit_start = 2
l1_fit_start = 4
l2_fit_start = 4
rk_fit_start = 2
markersize = 10
linewidth = 3
l0_L2_u_slope, l0_L2_u_fit = fit_power_law(Δts[l0_fit_start:end], l0_L2_u_errs[l0_fit_start:end])
l0_L2_w_slope, l0_L2_w_fit = fit_power_law(Δts[l0_fit_start:end], l0_L2_w_errs[l0_fit_start:end])
l0_L∞_u_slope, l0_L∞_u_fit = fit_power_law(Δts[l0_fit_start:end], l0_L∞_u_errs[l0_fit_start:end])
l0_L∞_w_slope, l0_L∞_w_fit = fit_power_law(Δts[l0_fit_start:end], l0_L∞_w_errs[l0_fit_start:end])
l1_L2_u_slope, l1_L2_u_fit = fit_power_law(Δts[l1_fit_start:end], l1_L2_u_errs[l1_fit_start:end])
l1_L2_w_slope, l1_L2_w_fit = fit_power_law(Δts[l1_fit_start:end], l1_L2_w_errs[l1_fit_start:end])
l1_L∞_u_slope, l1_L∞_u_fit = fit_power_law(Δts[l1_fit_start:end], l1_L∞_u_errs[l1_fit_start:end])
l1_L∞_w_slope, l1_L∞_w_fit = fit_power_law(Δts[l1_fit_start:end], l1_L∞_w_errs[l1_fit_start:end])
l2_L2_u_slope, l2_L2_u_fit = fit_power_law(Δts[l2_fit_start:end], l2_L2_u_errs[l2_fit_start:end])
l2_L2_w_slope, l2_L2_w_fit = fit_power_law(Δts[l2_fit_start:end], l2_L2_w_errs[l2_fit_start:end])
l2_L∞_u_slope, l2_L∞_u_fit = fit_power_law(Δts[l2_fit_start:end], l2_L∞_u_errs[l2_fit_start:end])
l2_L∞_w_slope, l2_L∞_w_fit = fit_power_law(Δts[l2_fit_start:end], l2_L∞_w_errs[l2_fit_start:end])

rk_L2_u_slope, rk_L2_u_fit = fit_power_law(Δts[rk_fit_start:end], rk_L2_u_errs[rk_fit_start:end])
rk_L2_w_slope, rk_L2_w_fit = fit_power_law(Δts[rk_fit_start:end], rk_L2_w_errs[rk_fit_start:end])
rk_L∞_u_slope, rk_L∞_u_fit = fit_power_law(Δts[rk_fit_start:end], rk_L∞_u_errs[rk_fit_start:end])
rk_L∞_w_slope, rk_L∞_w_fit = fit_power_law(Δts[rk_fit_start:end], rk_L∞_w_errs[rk_fit_start:end])

scatter!(axL2, Δts, l0_L2_u_errs, markersize=markersize)
scatter!(axL2, Δts, l0_L2_w_errs, markersize=markersize)
scatter!(axL2, Δts, l1_L2_u_errs, markersize=markersize)
scatter!(axL2, Δts, l1_L2_w_errs, markersize=markersize)
scatter!(axL2, Δts, l2_L2_u_errs, markersize=markersize)
scatter!(axL2, Δts, l2_L2_w_errs, markersize=markersize)
scatter!(axL2, Δts, rk_L2_u_errs, markersize=markersize)
scatter!(axL2, Δts, rk_L2_w_errs, markersize=markersize)

lines!(axL2, Δts[l0_fit_start:end], l0_L2_u_fit, label="FPJ0 u (slope = $(round(l0_L2_u_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL2, Δts[l0_fit_start:end], l0_L2_w_fit, label="FPJ0 w (slope = $(round(l0_L2_w_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL2, Δts[l1_fit_start:end], l1_L2_u_fit, label="FPJ1 u (slope = $(round(l1_L2_u_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL2, Δts[l1_fit_start:end], l1_L2_w_fit, label="FPJ1 w (slope = $(round(l1_L2_w_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL2, Δts[l2_fit_start:end], l2_L2_u_fit, label="FPJ2 u (slope = $(round(l2_L2_u_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL2, Δts[l2_fit_start:end], l2_L2_w_fit, label="FPJ2 w (slope = $(round(l2_L2_w_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL2, Δts[rk_fit_start:end], rk_L2_u_fit, label="RK3 u (slope = $(round(rk_L2_u_slope, sigdigits=3)))", linewidth=linewidth, linestyle=:dash)
lines!(axL2, Δts[rk_fit_start:end], rk_L2_w_fit, label="RK3 w (slope = $(round(rk_L2_w_slope, sigdigits=3)))", linewidth=linewidth, linestyle=:dash)

scatter!(axL∞, Δts, l0_L∞_u_errs, markersize=markersize)
scatter!(axL∞, Δts, l0_L∞_w_errs, markersize=markersize)
scatter!(axL∞, Δts, l1_L∞_u_errs, markersize=markersize)
scatter!(axL∞, Δts, l1_L∞_w_errs, markersize=markersize)
scatter!(axL∞, Δts, l2_L∞_u_errs, markersize=markersize)
scatter!(axL∞, Δts, l2_L∞_w_errs, markersize=markersize)
scatter!(axL∞, Δts, rk_L∞_u_errs, markersize=markersize)
scatter!(axL∞, Δts, rk_L∞_w_errs, markersize=markersize)
lines!(axL∞, Δts[l0_fit_start:end], l0_L∞_u_fit, label="FPJ0 u (slope = $(round(l0_L∞_u_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL∞, Δts[l0_fit_start:end], l0_L∞_w_fit, label="FPJ0 w (slope = $(round(l0_L∞_w_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL∞, Δts[l1_fit_start:end], l1_L∞_u_fit, label="FPJ1 u (slope = $(round(l1_L∞_u_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL∞, Δts[l1_fit_start:end], l1_L∞_w_fit, label="FPJ1 w (slope = $(round(l1_L∞_w_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL∞, Δts[l2_fit_start:end], l2_L∞_u_fit, label="FPJ2 u (slope = $(round(l2_L∞_u_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL∞, Δts[l2_fit_start:end], l2_L∞_w_fit, label="FPJ2 w (slope = $(round(l2_L∞_w_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL∞, Δts[rk_fit_start:end], rk_L∞_u_fit, label="RK3 u (slope = $(round(rk_L∞_u_slope, sigdigits=3)))", linewidth=linewidth, linestyle=:dash)
lines!(axL∞, Δts[rk_fit_start:end], rk_L∞_w_fit, label="RK3 w (slope = $(round(rk_L∞_w_slope, sigdigits=3)))", linewidth=linewidth, linestyle=:dash)
axislegend(axL2, position=:lt)
axislegend(axL∞, position=:lt)
display(fig)
# save(joinpath(OUTPUT_DIR, "$(filename_prefix)_convergence.png"), fig, px_per_unit=4)
# save(joinpath(OUTPUT_DIR, "$(filename_prefix)_convergence.pdf"), fig)

#%%
filename = joinpath(DATA_DIR, "$(filename_prefix)_RungeKutta3.jld2")
#%%
u_data = FieldTimeSeries(filename, "u")
w_data = FieldTimeSeries(filename, "w")
p_data = FieldTimeSeries(filename, "p")

times = u_data.times
Nt = length(times)
xC = xnodes(grid, Center())
xF = xnodes(grid, Face())
zC = znodes(grid, Center())
zF = znodes(grid, Face())

#%%
fig = Figure(size=(1200, 800), fontsize=18)

axu1 = Axis(fig[1, 1], xlabel="x", ylabel="z", title="Initial time, t = $(times[1])")
axw1 = Axis(fig[1, 2], xlabel="x", ylabel="z", title="Initial time, t = $(times[1])")
axp1 = Axis(fig[1, 3], xlabel="x", ylabel="z", title="Initial time, t = $(times[1])")

axu2 = Axis(fig[2, 1], xlabel="x", ylabel="z", title="Final time, t = $(times[end])")
axw2 = Axis(fig[2, 2], xlabel="x", ylabel="z", title="Final time, t = $(times[end])")
axp2 = Axis(fig[2, 3], xlabel="x", ylabel="z", title="Final time, t = $(times[end])")

u1 = interior(u_data[1], :, 1, :)
w1 = interior(w_data[1], :, 1, :)
p1 = interior(p_data[1], :, 1, :)

u2 = interior(u_data[end], :, 1, :)
w2 = interior(w_data[end], :, 1, :)
p2 = interior(p_data[end], :, 1, :)

ulim = (-maximum(abs.(vcat(u1, u2))) - 1e-4, maximum(abs.(vcat(u1, u2))) + 1e-4)
wlim = (-maximum(abs.(vcat(w1, w2))) - 1e-4, maximum(abs.(vcat(w1, w2))) + 1e-4)
plim = (minimum(vcat(p1, p2)) - 1e-4, maximum(vcat(p1, p2)) + 1e-4)

hmu1 = heatmap!(axu1, xF, zC, interior(u_data[1], :, 1, :), colormap=:balance, colorrange=ulim)
hmw1 = heatmap!(axw1, xC, zF, interior(w_data[1], :, 1, :), colormap=:balance, colorrange=wlim)
hmp1 = heatmap!(axp1, xC, zC, interior(p_data[1], :, 1, :), colormap=:viridis, colorrange=plim)

hmu2 = heatmap!(axu2, xF, zC, interior(u_data[end], :, 1, :), colormap=:balance, colorrange=ulim)
hmw2 = heatmap!(axw2, xC, zF, interior(w_data[end], :, 1, :), colormap=:balance, colorrange=wlim)
hmp2 = heatmap!(axp2, xC, zC, interior(p_data[end], :, 1, :), colormap=:viridis, colorrange=plim)

Colorbar(fig[3, 1], hmu1, vertical=false, flipaxis=false, label="y-averaged horizontal velocity (u)")
Colorbar(fig[3, 2], hmw1, vertical=false, flipaxis=false, label="y-averaged vertical velocity (w)")
Colorbar(fig[3, 3], hmp1, vertical=false, flipaxis=false, label="y-averaged pressure (p)")

linkaxes!(axu1, axu2, axw1, axw2, axp1, axp2)

hidexdecorations!(axu1, ticks=false)
hidexdecorations!(axw1, ticks=false)
hidexdecorations!(axp1, ticks=false)

hidexdecorations!(axu2, ticks=false, ticklabels=false, label=false)
hidexdecorations!(axw2, ticks=false, ticklabels=false, label=false)
hidexdecorations!(axp2, ticks=false, ticklabels=false, label=false)

hideydecorations!(axu1, ticks=false, ticklabels=false, label=false)
hideydecorations!(axw1, ticks=false)
hideydecorations!(axp1, ticks=false)

hideydecorations!(axu2, ticks=false, ticklabels=false, label=false)
hideydecorations!(axw2, ticks=false)
hideydecorations!(axp2, ticks=false)

# save(joinpath(OUTPUT_DIR, "$(filename_prefix)_initial_final.png"), fig, px_per_unit=4)
# save(joinpath(OUTPUT_DIR, "$(filename_prefix)_initial_final.pdf"), fig)

#%%
fig = Figure(size=(1200, 500))
n = Observable(1)
ax_u = Axis(fig[1, 1], title="Horizontal velocity", xlabel="x", ylabel="z")
ax_w = Axis(fig[1, 2], title="Vertical velocity", xlabel="x", ylabel="z")
ax_p = Axis(fig[1, 3], title="Pressure", xlabel="x", ylabel="z")

uₙ = @lift interior(u_data[$n], :, 1, :)
wₙ = @lift interior(w_data[$n], :, 1, :)
pₙ = @lift interior(p_data[$n], :, 1, :)

ulim = (-maximum(abs, interior(u_data)) - 1e-4, maximum(abs, interior(u_data)) + 1e-4)
wlim = (-maximum(abs, interior(w_data)) - 1e-4, maximum(abs, interior(w_data)) + 1e-4)
plim = (minimum(interior(p_data)) - 1e-4, maximum(interior(p_data)) + 1e-4)

hmu = heatmap!(ax_u, xF, zC, uₙ, colormap=:balance, colorrange=ulim)
hmw = heatmap!(ax_w, xC, zF, wₙ, colormap=:balance, colorrange=wlim)
hmp = heatmap!(ax_p, xC, zC, pₙ, colormap=:balance, colorrange=plim)

Colorbar(fig[2, 1], hmu, vertical=false, flipaxis=false, label="u")
Colorbar(fig[2, 2], hmw, vertical=false, flipaxis=false, label="w")
Colorbar(fig[2, 3], hmp, vertical=false, flipaxis=false, label="p")

titlestr = @lift "Time = $(round(times[$n], sigdigits=2)) s"
Label(fig[0, :], titlestr, font=:bold)
display(fig)

# CairoMakie.record(fig, joinpath(OUTPUT_DIR, "$(filename_prefix)_RK3.mp4"), 1:Nt, framerate=15) do nn
#     n[] = nn
# end

#%%
wall_time_l0, wall_time_l1, wall_time_l2, wall_time_rk = jldopen(joinpath(DATA_DIR, "$(filename_prefix)_timings.jld2"), "r") do file
    file["l0"], file["l1"], file["l2"], file["rk"]
end

# Regular-domain timing stores one sample vector per method.
median_wall_time_l0 = median(wall_time_l0)
median_wall_time_l1 = median(wall_time_l1)
median_wall_time_l2 = median(wall_time_l2)
median_wall_time_rk = median(wall_time_rk)

Nts = stop_time ./ Δts

total_time_l0 = median_wall_time_l0 .* Nts
total_time_l1 = median_wall_time_l1 .* Nts
total_time_l2 = median_wall_time_l2 .* Nts
total_time_rk = median_wall_time_rk .* Nts

# jldopen(joinpath(DATA_DIR, "$(filename_prefix)_timings.jld2"), "w") do file
#     file["l0"] = wall_time_l0
#     file["l1"] = wall_time_l1
#     file["l2"] = wall_time_l2
#     file["rk"] = wall_time_rk
# end
#%%
fig = Figure(size=(900, 500), fontsize=15)
axL2 = Axis(fig[1, 1], title="L2 error at final time", xlabel="Wall time (s)", ylabel="L2 error", xscale=log10, yscale=log10)
axL∞ = Axis(fig[1, 2], title="L∞ error at final time", xlabel="Wall time (s)", ylabel="L∞ error", xscale=log10, yscale=log10)

lm_fit_start = 3
markersize = 10
linewidth = 3
l0_L2_u_slope, l0_L2_u_fit = fit_power_law(total_time_l0[lm_fit_start:end], l0_L2_u_errs[lm_fit_start:end])
l0_L2_w_slope, l0_L2_w_fit = fit_power_law(total_time_l0[lm_fit_start:end], l0_L2_w_errs[lm_fit_start:end])
l0_L∞_u_slope, l0_L∞_u_fit = fit_power_law(total_time_l0[lm_fit_start:end], l0_L∞_u_errs[lm_fit_start:end])
l0_L∞_w_slope, l0_L∞_w_fit = fit_power_law(total_time_l0[lm_fit_start:end], l0_L∞_w_errs[lm_fit_start:end])
l1_L2_u_slope, l1_L2_u_fit = fit_power_law(total_time_l1[lm_fit_start:end], l1_L2_u_errs[lm_fit_start:end])
l1_L2_w_slope, l1_L2_w_fit = fit_power_law(total_time_l1[lm_fit_start:end], l1_L2_w_errs[lm_fit_start:end])
l1_L∞_u_slope, l1_L∞_u_fit = fit_power_law(total_time_l1[lm_fit_start:end], l1_L∞_u_errs[lm_fit_start:end])
l1_L∞_w_slope, l1_L∞_w_fit = fit_power_law(total_time_l1[lm_fit_start:end], l1_L∞_w_errs[lm_fit_start:end])
l2_L2_u_slope, l2_L2_u_fit = fit_power_law(total_time_l2[lm_fit_start:end], l2_L2_u_errs[lm_fit_start:end])
l2_L2_w_slope, l2_L2_w_fit = fit_power_law(total_time_l2[lm_fit_start:end], l2_L2_w_errs[lm_fit_start:end])
l2_L∞_u_slope, l2_L∞_u_fit = fit_power_law(total_time_l2[lm_fit_start:end], l2_L∞_u_errs[lm_fit_start:end])
l2_L∞_w_slope, l2_L∞_w_fit = fit_power_law(total_time_l2[lm_fit_start:end], l2_L∞_w_errs[lm_fit_start:end])

rk_L2_u_slope, rk_L2_u_fit = fit_power_law(total_time_rk, rk_L2_u_errs)
rk_L2_w_slope, rk_L2_w_fit = fit_power_law(total_time_rk, rk_L2_w_errs)
rk_L∞_u_slope, rk_L∞_u_fit = fit_power_law(total_time_rk, rk_L∞_u_errs)
rk_L∞_w_slope, rk_L∞_w_fit = fit_power_law(total_time_rk, rk_L∞_w_errs)

scatter!(axL2, total_time_l0, l0_L2_u_errs, markersize=markersize)
scatter!(axL2, total_time_l0, l0_L2_w_errs, markersize=markersize)
scatter!(axL2, total_time_l1, l1_L2_u_errs, markersize=markersize)
scatter!(axL2, total_time_l1, l1_L2_w_errs, markersize=markersize)
scatter!(axL2, total_time_l2, l2_L2_u_errs, markersize=markersize)
scatter!(axL2, total_time_l2, l2_L2_w_errs, markersize=markersize)
scatter!(axL2, total_time_rk, rk_L2_u_errs, markersize=markersize)
scatter!(axL2, total_time_rk, rk_L2_w_errs, markersize=markersize)

lines!(axL2, total_time_l0[lm_fit_start:end], l0_L2_u_fit, label="FPJ0-RK3 u (slope = $(round(l0_L2_u_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL2, total_time_l0[lm_fit_start:end], l0_L2_w_fit, label="FPJ0-RK3 w (slope = $(round(l0_L2_w_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL2, total_time_l1[lm_fit_start:end], l1_L2_u_fit, label="FPJ1-RK3 u (slope = $(round(l1_L2_u_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL2, total_time_l1[lm_fit_start:end], l1_L2_w_fit, label="FPJ1-RK3 w (slope = $(round(l1_L2_w_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL2, total_time_l2[lm_fit_start:end], l2_L2_u_fit, label="FPJ2-RK3 u (slope = $(round(l2_L2_u_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL2, total_time_l2[lm_fit_start:end], l2_L2_w_fit, label="FPJ2-RK3 w (slope = $(round(l2_L2_w_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL2, total_time_rk, rk_L2_u_fit, label="RK3 u (slope = $(round(rk_L2_u_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL2, total_time_rk, rk_L2_w_fit, label="RK3 w (slope = $(round(rk_L2_w_slope, sigdigits=3)))", linewidth=linewidth)

scatter!(axL∞, total_time_l0, l0_L∞_u_errs, markersize=markersize)
scatter!(axL∞, total_time_l0, l0_L∞_w_errs, markersize=markersize)
scatter!(axL∞, total_time_l1, l1_L∞_u_errs, markersize=markersize)
scatter!(axL∞, total_time_l1, l1_L∞_w_errs, markersize=markersize)
scatter!(axL∞, total_time_l2, l2_L∞_u_errs, markersize=markersize)
scatter!(axL∞, total_time_l2, l2_L∞_w_errs, markersize=markersize)
scatter!(axL∞, total_time_rk, rk_L∞_u_errs, markersize=markersize)
scatter!(axL∞, total_time_rk, rk_L∞_w_errs, markersize=markersize)
lines!(axL∞, total_time_l0[lm_fit_start:end], l0_L∞_u_fit, label="FPJ0-RK3 u (slope = $(round(l0_L∞_u_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL∞, total_time_l0[lm_fit_start:end], l0_L∞_w_fit, label="FPJ0-RK3 w (slope = $(round(l0_L∞_w_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL∞, total_time_l1[lm_fit_start:end], l1_L∞_u_fit, label="FPJ1-RK3 u (slope = $(round(l1_L∞_u_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL∞, total_time_l1[lm_fit_start:end], l1_L∞_w_fit, label="FPJ1-RK3 w (slope = $(round(l1_L∞_w_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL∞, total_time_l2[lm_fit_start:end], l2_L∞_u_fit, label="FPJ2-RK3 u (slope = $(round(l2_L∞_u_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL∞, total_time_l2[lm_fit_start:end], l2_L∞_w_fit, label="FPJ2-RK3 w (slope = $(round(l2_L∞_w_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL∞, total_time_rk, rk_L∞_u_fit, label="RK3 u (slope = $(round(rk_L∞_u_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL∞, total_time_rk, rk_L∞_w_fit, label="RK3 w (slope = $(round(rk_L∞_w_slope, sigdigits=3)))", linewidth=linewidth)
axislegend(axL2, position=:rt)
axislegend(axL∞, position=:rt)
display(fig)
# save("./$(filename_prefix)_convergence_wall_time.png", fig, px_per_unit=4)
# save("./$(filename_prefix)_convergence_wall_time.pdf", fig)
#%%
fig = Figure(size=(900, 500), fontsize=15)
axL2 = Axis(fig[1, 1], xlabel="Wall time (s)", ylabel="L2 error", xscale=log10, yscale=log10)
axL∞ = Axis(fig[1, 2], xlabel="Wall time (s)", ylabel="L∞ error", xscale=log10, yscale=log10)

l0_fit_start = 5
l1_fit_start = 7
l2_fit_start = 7
rk_fit_start = 4
markersize = 13
linewidth = 2

l0_L2_slope, l0_L2_fit = fit_power_law(total_time_l0[l0_fit_start:end], l0_L2_errs[l0_fit_start:end])
l0_L∞_slope, l0_L∞_fit = fit_power_law(total_time_l0[l0_fit_start:end], l0_L∞_errs[l0_fit_start:end])
l1_L2_slope, l1_L2_fit = fit_power_law(total_time_l1[l1_fit_start:end], l1_L2_errs[l1_fit_start:end])
l1_L∞_slope, l1_L∞_fit = fit_power_law(total_time_l1[l1_fit_start:end], l1_L∞_errs[l1_fit_start:end])
l2_L2_slope, l2_L2_fit = fit_power_law(total_time_l2[l2_fit_start:end], l2_L2_errs[l2_fit_start:end])
l2_L∞_slope, l2_L∞_fit = fit_power_law(total_time_l2[l2_fit_start:end], l2_L∞_errs[l2_fit_start:end])
rk_L2_slope, rk_L2_fit = fit_power_law(total_time_rk[rk_fit_start:end], rk_L2_errs[rk_fit_start:end])
rk_L∞_slope, rk_L∞_fit = fit_power_law(total_time_rk[rk_fit_start:end], rk_L∞_errs[rk_fit_start:end])

scatter!(axL2, total_time_l0, l0_L2_errs, markersize=markersize, marker=:circle)
scatter!(axL2, total_time_l1, l1_L2_errs, markersize=markersize, marker=:rect)
scatter!(axL2, total_time_l2, l2_L2_errs, markersize=markersize, marker=:utriangle)
scatter!(axL2, total_time_rk, rk_L2_errs, markersize=markersize, marker=:cross)

lines!(axL2, total_time_l0[l0_fit_start:end], l0_L2_fit, label="FPJ0 (slope = $(round(l0_L2_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL2, total_time_l1[l1_fit_start:end], l1_L2_fit, label="FPJ1 (slope = $(round(l1_L2_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL2, total_time_l2[l2_fit_start:end], l2_L2_fit, label="FPJ2 (slope = $(round(l2_L2_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL2, total_time_rk[rk_fit_start:end], rk_L2_fit, label="RK3 (slope = $(round(rk_L2_slope, sigdigits=3)))", linewidth=linewidth, linestyle=:dash)

scatter!(axL∞, total_time_l0, l0_L∞_errs, markersize=markersize, marker=:circle)
scatter!(axL∞, total_time_l1, l1_L∞_errs, markersize=markersize, marker=:rect)
scatter!(axL∞, total_time_l2, l2_L∞_errs, markersize=markersize, marker=:utriangle)
scatter!(axL∞, total_time_rk, rk_L∞_errs, markersize=markersize, marker=:cross)

lines!(axL∞, total_time_l0[l0_fit_start:end], l0_L∞_fit, label="FPJ0 (slope = $(round(l0_L∞_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL∞, total_time_l1[l1_fit_start:end], l1_L∞_fit, label="FPJ1 (slope = $(round(l1_L∞_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL∞, total_time_l2[l2_fit_start:end], l2_L∞_fit, label="FPJ2 (slope = $(round(l2_L∞_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL∞, total_time_rk[rk_fit_start:end], rk_L∞_fit, label="RK3 (slope = $(round(rk_L∞_slope, sigdigits=3)))", linewidth=linewidth, linestyle=:dash)

axislegend(axL2, position=:rt)
axislegend(axL∞, position=:rt)
display(fig)
# save(joinpath(OUTPUT_DIR, "$(filename_prefix)_combined_error_vs_runtime.png"), fig, px_per_unit=4)
# save(joinpath(OUTPUT_DIR, "$(filename_prefix)_combined_error_vs_runtime.pdf"), fig)




