using Oceananigans
using Oceananigans.Units
using Printf
using JLD2
using Statistics
using Oceananigans.Models.NonhydrostaticModels: ConjugateGradientPoissonSolver, FFTBasedPoissonSolver
using Oceananigans.Utils: launch!
using Oceananigans.Architectures: architecture
using Oceananigans.Operators
using KernelAbstractions: @kernel, @index
using CUDA
using CairoMakie
using BenchmarkTools
using NaNStatistics
using Oceananigans.ImmersedBoundaries: mask_immersed_field!

function initial_conditions!(model)
    δx = model.grid.underlying_grid.Δxᶜᵃᵃ
    U = 1
    U_bg = 0

    R = 0.1      # width/radius of the broad center
    δ = 3δx      # dropoff thickness; smaller = sharper

    z₀ = 0.65

    r(x, y, z) = sqrt((z - z₀)^2)

    uᵢ(x, y, z) = U_bg + (U - U_bg) * 0.5 * (1 - tanh((r(x, y, z) - R) / δ))

    set!(model, u=uᵢ)
end

function setup_grid(Nx, Ny, Nz)
    grid = RectilinearGrid(GPU(), Float64,
                           size = (Nx, Ny, Nz),
                           halo = (6, 6, 6),
                           x = (0, 1),
                           y = (0, 1),
                           z = (0, 1),
                           topology = (Bounded, Periodic, Bounded))

    staircase(x, y) = (5 + tanh(40*(x - 1/6)) +
                           tanh(40*(x - 2/6)) +
                           tanh(40*(x - 3/6)) +
                           tanh(40*(x - 4/6)) +
                           tanh(40*(x - 5/6))) / 10

    grid = ImmersedBoundaryGrid(grid, GridFittedBottom(staircase))

    return grid
end

function setup_model(grid, timestepper=:RungeKutta3)
    preconditioner = FFTBasedPoissonSolver(grid.underlying_grid)
    pressure_solver = ConjugateGradientPoissonSolver(grid, maxiter=30; preconditioner)

    closure = ScalarDiffusivity(ν=1e-5, κ=1e-5)

    model = NonhydrostaticModel(grid;
                                # advection = WENO(order=9),
                                advection = UpwindBiased(order=advection_order),
                                timestepper,
                                pressure_solver,
                                closure)

    initial_conditions!(model)
    return model
end

Nx = Ny = Nz = 256
Δx = 1 / Nx
advection_order = 9
filename_prefix = "staircase_shear_flow_upwindbiased$(advection_order)"

grid = setup_grid(Nx, Ny, Nz)

# cfls = [0.8, 0.7, 0.4, 0.3, 0.2, 0.1, 0.05]
# cfls = [0.8, 0.4, 0.3, 0.2, 0.1]
# cfls = [0.6, 0.5]
# cfls = [0.25]
# cfls = [0.05]

cfls = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.25, 0.1]
Δts = cfls .* Δx ./ 1.7

cfl_ref = 0.05
Δt_ref = cfl_ref * Δx / 1.7
# Δt_ref = Δts[end] / 4

stop_time = 10

# timestepper = :RungeKutta3
# timestepper = :LeMoinRungeKutta3
timesteppers = [:RungeKutta3, :LeMoinRungeKutta3FPJ0, :LeMoinRungeKutta3FPJ1, :LeMoinRungeKutta3FPJ2]
# timesteppers = [:RungeKutta3, :LeMoinRungeKutta3FPJ0]
# timesteppers = [:LeMoinRungeKutta3FPJ1, :LeMoinRungeKutta3FPJ2]
# timesteppers = [:RungeKutta3]
# timesteppers = [:LeMoinRungeKutta3FPJ0]
# timesteppers = [:LeMoinRungeKutta3FPJ1]
# timesteppers = [:LeMoinRungeKutta3FPJ2]
# timesteppers = [:RungeKutta3, :LeMoinRungeKutta3FPJ1, :LeMoinRungeKutta3FPJ2]

const DATA_DIR = joinpath(@__DIR__, "data")
const OUTPUT_DIR = joinpath(@__DIR__, "output")
mkpath(DATA_DIR)
mkpath(OUTPUT_DIR)

function run_simulation(; timestepper, Δt, filename)
    grid = setup_grid(Nx, Ny, Nz)

    model = setup_model(grid, timestepper)

    # stop_iteration = 100
    # simulation = Simulation(model; Δt=1e-4, stop_iteration=100, minimum_relative_step = 1e-10)
    simulation = Simulation(model; Δt, stop_time, minimum_relative_step = 1e-10)

    Nt = round(Int, stop_time / Δt)

    # time_wizard = TimeStepWizard(cfl=0.3, max_change=1.05, min_Δt=1e-5, max_Δt=10)
    # simulation.callbacks[:wizard] = Callback(time_wizard, IterationInterval(1))

    u, v, w = model.velocities

    d = CenterField(grid)

    @kernel function _divergence!(target_field, u, v, w, grid)
        i, j, k = @index(Global, NTuple)
        @inbounds target_field[i, j, k] = divᶜᶜᶜ(i, j, k, grid, u, v, w)
    end

    function compute_flow_divergence!(target_field, model)
        grid = model.grid
        u, v, w = model.velocities
        arch = architecture(grid)
        launch!(arch, grid, :xyz, _divergence!, target_field, u, v, w, grid)
        return nothing
    end

    wall_time = Ref(time_ns())

    function progress(sim)
        pressure_solver = sim.model.pressure_solver

        if pressure_solver isa ConjugateGradientPoissonSolver
            pressure_iters = iteration(pressure_solver)
        else
            pressure_iters = 0
        end

        msg = @sprintf("Iter: %d, time: %s, Δt: %6.3e, Poisson iters: %d",
                        iteration(sim), prettytime(time(sim)), sim.Δt, pressure_iters)

        elapsed = 1e-9 * (time_ns() - wall_time[])

        compute_flow_divergence!(d, sim.model)

        msg *= @sprintf(", max u: %6.3e, max v: %6.3e, max w: %6.3e, max d: %6.3e, max pressure: %6.3e, wall time: %s",
                        maximum(abs, sim.model.velocities.u),
                        maximum(abs, sim.model.velocities.v),
                        maximum(abs, sim.model.velocities.w),
                        maximum(abs, d),
                        maximum(abs, sim.model.pressures.pNHS),
                        prettytime(elapsed))

        @info msg
        wall_time[] = time_ns()

        return nothing
    end

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(max(1, Nt ÷ 5)))

    outputs = merge(model.velocities, (; p=model.pressures.pNHS, d))

    simulation.output_writers[:jld2] = JLD2Writer(model, outputs;
                                                  dir = DATA_DIR,
                                                  filename = filename,
                                                  schedule = TimeInterval(10),
                                                  overwrite_existing = true)

    run!(simulation)
end

#%%
for ts in timesteppers, Δt in Δts
    filename = "$(filename_prefix)_$(ts)_Nx$(Nx)_Δt$(Δt)_2.jld2"
    @info "Running $ts with Δt = $Δt"
    run_simulation(timestepper=ts, Δt=Δt, filename=filename)
end

filename_ref = "$(filename_prefix)_RungeKutta3_Nx$(Nx)_Δt$(Δt_ref).jld2"
@info "Running REFERENCE RungeKutta3 with Δt = $Δt_ref"
run_simulation(timestepper=:RungeKutta3, Δt=Δt_ref, filename=filename_ref)

#%%
rk_filenames = ["$(filename_prefix)_RungeKutta3_Nx$(Nx)_Δt$(Δt).jld2" for Δt in Δts]
l0_filenames = ["$(filename_prefix)_LeMoinRungeKutta3FPJ0_Nx$(Nx)_Δt$(Δt).jld2" for Δt in Δts]
l1_filenames = ["$(filename_prefix)_LeMoinRungeKutta3FPJ1_Nx$(Nx)_Δt$(Δt).jld2" for Δt in Δts]
l2_filenames = ["$(filename_prefix)_LeMoinRungeKutta3FPJ2_Nx$(Nx)_Δt$(Δt).jld2" for Δt in Δts]
ref_filename = "$(filename_prefix)_RungeKutta3_Nx$(Nx)_Δt$(Δt_ref)_2.jld2"

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

times = ref_u_data.times

for i in 1:length(rk_u_datas)
    for j in 1:length(rk_u_datas[i])
        mask_immersed_field!(rk_u_datas[i][j], NaN)
        mask_immersed_field!(rk_w_datas[i][j], NaN)
    end

    for j in 1:length(l0_u_datas[i])
        mask_immersed_field!(l0_u_datas[i][j], NaN)
        mask_immersed_field!(l0_w_datas[i][j], NaN)
    end
    for j in 1:length(l1_u_datas[i])
        mask_immersed_field!(l1_u_datas[i][j], NaN)
        mask_immersed_field!(l1_w_datas[i][j], NaN)
    end
    for j in 1:length(l2_u_datas[i])
        mask_immersed_field!(l2_u_datas[i][j], NaN)
        mask_immersed_field!(l2_w_datas[i][j], NaN)
    end
end

for j in 1:length(times)
    mask_immersed_field!(ref_u_data[j], NaN)
    mask_immersed_field!(ref_w_data[j], NaN)
end

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

        L2_u = sqrt(nanmean(eu.^2))
        L2_w = sqrt(nanmean(ew.^2))

        L∞_u = nanmaximum(abs.(eu))
        L∞_w = nanmaximum(abs.(ew))

        return (; L∞_u, L∞_w,
                L2_u, L2_w,
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
cfls_for_ticks = [0.8, 0.6, 0.4, 0.3, 0.2, 0.1]
Δts_for_ticks = cfls_for_ticks .* Δx ./ 1.7
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

l0_L2_fit_start = 5
l1_L2_fit_start = 6
l2_L2_fit_start = 6
rk_L2_fit_start = 5

l0_L∞_fit_start = 7
l1_L∞_fit_start = 7
l2_L∞_fit_start = 6
rk_L∞_fit_start = 6

markersize = 13
linewidth = 2

l0_L2_slope, l0_L2_fit = fit_power_law(Δts[l0_L2_fit_start:end], l0_L2_errs[l0_L2_fit_start:end])
l0_L∞_slope, l0_L∞_fit = fit_power_law(Δts[l0_L∞_fit_start:end], l0_L∞_errs[l0_L∞_fit_start:end])
l1_L2_slope, l1_L2_fit = fit_power_law(Δts[l1_L2_fit_start:end], l1_L2_errs[l1_L2_fit_start:end])
l1_L∞_slope, l1_L∞_fit = fit_power_law(Δts[l1_L∞_fit_start:end], l1_L∞_errs[l1_L∞_fit_start:end])
l2_L2_slope, l2_L2_fit = fit_power_law(Δts[l2_L2_fit_start:end], l2_L2_errs[l2_L2_fit_start:end])
l2_L∞_slope, l2_L∞_fit = fit_power_law(Δts[l2_L∞_fit_start:end], l2_L∞_errs[l2_L∞_fit_start:end])
rk_L2_slope, rk_L2_fit = fit_power_law(Δts[rk_L2_fit_start:end], rk_L2_errs[rk_L2_fit_start:end])
rk_L∞_slope, rk_L∞_fit = fit_power_law(Δts[rk_L∞_fit_start:end], rk_L∞_errs[rk_L∞_fit_start:end])

scatter!(axL2, Δts, l0_L2_errs, markersize=markersize, marker=:circle)
scatter!(axL2, Δts, l1_L2_errs, markersize=markersize, marker=:rect)
scatter!(axL2, Δts, l2_L2_errs, markersize=markersize, marker=:utriangle)
scatter!(axL2, Δts, rk_L2_errs, markersize=markersize, marker=:cross)

lines!(axL2, Δts[l0_L2_fit_start:end], l0_L2_fit, label="FPJ0 (slope = $(round(l0_L2_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL2, Δts[l1_L2_fit_start:end], l1_L2_fit, label="FPJ1 (slope = $(round(l1_L2_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL2, Δts[l2_L2_fit_start:end], l2_L2_fit, label="FPJ2 (slope = $(round(l2_L2_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL2, Δts[rk_L2_fit_start:end], rk_L2_fit, label="RK3 (slope = $(round(rk_L2_slope, sigdigits=3)))", linewidth=linewidth)

scatter!(axL∞, Δts, l0_L∞_errs, markersize=markersize, marker=:circle)
scatter!(axL∞, Δts, l1_L∞_errs, markersize=markersize, marker=:rect)
scatter!(axL∞, Δts, l2_L∞_errs, markersize=markersize, marker=:utriangle)
scatter!(axL∞, Δts, rk_L∞_errs, markersize=markersize, marker=:cross)

lines!(axL∞, Δts[l0_L∞_fit_start:end], l0_L∞_fit, label="FPJ0 (slope = $(round(l0_L∞_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL∞, Δts[l1_L∞_fit_start:end], l1_L∞_fit, label="FPJ1 (slope = $(round(l1_L∞_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL∞, Δts[l2_L∞_fit_start:end], l2_L∞_fit, label="FPJ2 (slope = $(round(l2_L∞_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL∞, Δts[rk_L∞_fit_start:end], rk_L∞_fit, label="RK3 (slope = $(round(rk_L∞_slope, sigdigits=3)))", linewidth=linewidth)

axislegend(axL2, position=:rb)
axislegend(axL∞, position=:rb)
display(fig)
save(joinpath(OUTPUT_DIR, "$(filename_prefix)_combined_error_convergence.png"), fig, px_per_unit=4)
save(joinpath(OUTPUT_DIR, "$(filename_prefix)_combined_error_convergence.pdf"), fig)

#%%
model = setup_model(grid, :RungeKutta3)
Δt = 0.5 * Δx / 10

simulation = Simulation(model; Δt, stop_time, minimum_relative_step = 1e-10)

time_wizard = TimeStepWizard(cfl=0.6, max_change=1.05, min_Δt=1e-5, max_Δt=10)
simulation.callbacks[:wizard] = Callback(time_wizard, IterationInterval(1))

u, v, w = model.velocities

d = CenterField(grid)

@kernel function _divergence!(target_field, u, v, w, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds target_field[i, j, k] = divᶜᶜᶜ(i, j, k, grid, u, v, w)
end

function compute_flow_divergence!(target_field, model)
    grid = model.grid
    u, v, w = model.velocities
    arch = architecture(grid)
    launch!(arch, grid, :xyz, _divergence!, target_field, u, v, w, grid)
    return nothing
end

wall_time = Ref(time_ns())

function progress(sim)
    pressure_solver = sim.model.pressure_solver

    if pressure_solver isa ConjugateGradientPoissonSolver
        pressure_iters = iteration(pressure_solver)
    else
        pressure_iters = 0
    end

    msg = @sprintf("Iter: %d, time: %s, Δt: %6.3e, Poisson iters: %d",
                    iteration(sim), prettytime(time(sim)), sim.Δt, pressure_iters)

    elapsed = 1e-9 * (time_ns() - wall_time[])

    compute_flow_divergence!(d, sim.model)

    msg *= @sprintf(", max u: %6.3e, max v: %6.3e, max w: %6.3e, max d: %6.3e, max pressure: %6.3e, wall time: %s",
                    maximum(abs, sim.model.velocities.u),
                    maximum(abs, sim.model.velocities.v),
                    maximum(abs, sim.model.velocities.w),
                    maximum(abs, d),
                    maximum(abs, sim.model.pressures.pNHS),
                    prettytime(elapsed))

    @info msg
    wall_time[] = time_ns()

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

u, v, w = model.velocities
ubar = Average(u, dims=2)
vbar = Average(v, dims=2)
wbar = Average(w, dims=2)
pbar = Average(model.pressures.pNHS, dims=2)
dbar = Average(d, dims=2)

# outputs = merge(model.velocities, (; p=model.pressures.pNHS, d))
outputs = (; u=ubar, w=wbar, p=pbar, d=dbar)

filename = joinpath(DATA_DIR, "$(filename_prefix)_RungeKutta3.jld2")
simulation.output_writers[:jld2] = JLD2Writer(model, outputs;
                                            filename = filename,
                                            schedule = TimeInterval(0.1),
                                            overwrite_existing = true)

run!(simulation)
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

for i in 1:length(times)
    mask_immersed_field!(u_data[i], NaN)
    mask_immersed_field!(w_data[i], NaN)
    mask_immersed_field!(p_data[i], NaN)
end
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

ulim = (-nanmaximum(abs.(vcat(u1, u2))) - 1e-4, nanmaximum(abs.(vcat(u1, u2))) + 1e-4)
wlim = (-nanmaximum(abs.(vcat(w1, w2))) - 1e-4, nanmaximum(abs.(vcat(w1, w2))) + 1e-4)
plim = (nanminimum(vcat(p1, p2)) - 1e-4, nanmaximum(vcat(p1, p2)) + 1e-4)

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

save(joinpath(OUTPUT_DIR, "$(filename_prefix)_initial_final.png"), fig, px_per_unit=4)
save(joinpath(OUTPUT_DIR, "$(filename_prefix)_initial_final.pdf"), fig)
#%%
fig = Figure(size=(1200, 500))
n = Observable(1)
ax_u = Axis(fig[1, 1], title="Horizontal velocity", xlabel="x", ylabel="z")
ax_w = Axis(fig[1, 2], title="Vertical velocity", xlabel="x", ylabel="z")
ax_p = Axis(fig[1, 3], title="Pressure", xlabel="x", ylabel="z")

uₙ = @lift interior(u_data[$n], :, 1, :)
wₙ = @lift interior(w_data[$n], :, 1, :)
pₙ = @lift interior(p_data[$n], :, 1, :)

ulim = (-nanmaximum(abs.(interior(u_data))) - 1e-4, nanmaximum(abs.(interior(u_data))) + 1e-4)
wlim = (-nanmaximum(abs.(interior(w_data))) - 1e-4, nanmaximum(abs.(interior(w_data))) + 1e-4)
plim = (nanminimum(interior(p_data)) - 1e-4, nanmaximum(interior(p_data)) + 1e-4)

hmu = heatmap!(ax_u, xF, zC, uₙ, colormap=:balance, colorrange=ulim)
hmw = heatmap!(ax_w, xC, zF, wₙ, colormap=:balance, colorrange=wlim)
hmp = heatmap!(ax_p, xC, zC, pₙ, colormap=:balance, colorrange=plim)

Colorbar(fig[2, 1], hmu, vertical=false, flipaxis=false, label="u")
Colorbar(fig[2, 2], hmw, vertical=false, flipaxis=false, label="w")
Colorbar(fig[2, 3], hmp, vertical=false, flipaxis=false, label="p")

titlestr = @lift "Time = $(round(times[$n], sigdigits=2)) s"
Label(fig[0, :], titlestr, font=:bold)
display(fig)

CairoMakie.record(fig, joinpath(OUTPUT_DIR, "$(filename_prefix)_RK3.mp4"), 1:Nt, framerate=15) do nn
    n[] = nn
end

#%%
ntrials = 200

# wall_time_l0 = zeros(length(Δts), ntrials)
# wall_time_l1 = zeros(length(Δts), ntrials)
# wall_time_l2 = zeros(length(Δts), ntrials)
# wall_time_rk = zeros(length(Δts), ntrials)

# for (n, Δt) in enumerate(Δts)
#     @info "Timing with Δt = $Δt"
#     model_l0 = setup_model(grid, :LeMoinRungeKutta3FPJ0)
#     model_l1 = setup_model(grid, :LeMoinRungeKutta3FPJ1)
#     model_l2 = setup_model(grid, :LeMoinRungeKutta3FPJ2)
#     model_rk = setup_model(grid, :RungeKutta3)

#     for i in 1:ntrials
#         t_l0 = @timed time_step!(model_l0, Δt)
#         t_l1 = @timed time_step!(model_l1, Δt)
#         t_l2 = @timed time_step!(model_l2, Δt)
#         t_rk = @timed time_step!(model_rk, Δt)
#         wall_time_l0[n, i] = t_l0.time
#         wall_time_l1[n, i] = t_l1.time
#         wall_time_l2[n, i] = t_l2.time
#         wall_time_rk[n, i] = t_rk.time
#     end
# end

# jldopen(joinpath(DATA_DIR, "$(filename_prefix)_timings.jld2"), "w") do file
#     file["l0"] = wall_time_l0
#     file["l1"] = wall_time_l1
#     file["l2"] = wall_time_l2
#     file["rk"] = wall_time_rk
# end

wall_time_l0, wall_time_l1, wall_time_l2, wall_time_rk = jldopen(joinpath(DATA_DIR, "$(filename_prefix)_timings.jld2"), "r") do file
    file["l0"], file["l1"], file["l2"], file["rk"]
end

median_wall_time_l0 = vec(median(wall_time_l0, dims=2))
median_wall_time_l1 = vec(median(wall_time_l1, dims=2))
median_wall_time_l2 = vec(median(wall_time_l2, dims=2))
median_wall_time_rk = vec(median(wall_time_rk, dims=2))

Nts = stop_time ./ Δts

total_time_l0 = median_wall_time_l0 .* Nts
total_time_l1 = median_wall_time_l1 .* Nts
total_time_l2 = median_wall_time_l2 .* Nts
total_time_rk = median_wall_time_rk .* Nts

#%%
fig = Figure(size=(900, 500), fontsize=15)
axL2 = Axis(fig[1, 1], xlabel="Wall time (s)", ylabel="L2 error", xscale=log10, yscale=log10)
axL∞ = Axis(fig[1, 2], xlabel="Wall time (s)", ylabel="L∞ error", xscale=log10, yscale=log10)

l0_L2_fit_start = 5
l1_L2_fit_start = 6
l2_L2_fit_start = 6
rk_L2_fit_start = 2

l0_L∞_fit_start = 7
l1_L∞_fit_start = 7
l2_L∞_fit_start = 6
rk_L∞_fit_start = 6
markersize = 13
linewidth = 2

l0_L2_slope, l0_L2_fit = fit_power_law(total_time_l0[l0_L2_fit_start:end], l0_L2_errs[l0_L2_fit_start:end])
l0_L∞_slope, l0_L∞_fit = fit_power_law(total_time_l0[l0_L∞_fit_start:end], l0_L∞_errs[l0_L∞_fit_start:end])
l1_L2_slope, l1_L2_fit = fit_power_law(total_time_l1[l1_L2_fit_start:end], l1_L2_errs[l1_L2_fit_start:end])
l1_L∞_slope, l1_L∞_fit = fit_power_law(total_time_l1[l1_L∞_fit_start:end], l1_L∞_errs[l1_L∞_fit_start:end])
l2_L2_slope, l2_L2_fit = fit_power_law(total_time_l2[l2_L2_fit_start:end], l2_L2_errs[l2_L2_fit_start:end])
l2_L∞_slope, l2_L∞_fit = fit_power_law(total_time_l2[l2_L∞_fit_start:end], l2_L∞_errs[l2_L∞_fit_start:end])
rk_L2_slope, rk_L2_fit = fit_power_law(total_time_rk[rk_L2_fit_start:end], rk_L2_errs[rk_L2_fit_start:end])
rk_L∞_slope, rk_L∞_fit = fit_power_law(total_time_rk[rk_L∞_fit_start:end], rk_L∞_errs[rk_L∞_fit_start:end])

scatter!(axL2, total_time_l0, l0_L2_errs, markersize=markersize, marker=:circle)
scatter!(axL2, total_time_l1, l1_L2_errs, markersize=markersize, marker=:rect)
scatter!(axL2, total_time_l2, l2_L2_errs, markersize=markersize, marker=:utriangle)
scatter!(axL2, total_time_rk, rk_L2_errs, markersize=markersize, marker=:cross)

lines!(axL2, total_time_l0[l0_L2_fit_start:end], l0_L2_fit, label="FPJ0 (slope = $(round(l0_L2_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL2, total_time_l1[l1_L2_fit_start:end], l1_L2_fit, label="FPJ1 (slope = $(round(l1_L2_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL2, total_time_l2[l2_L2_fit_start:end], l2_L2_fit, label="FPJ2 (slope = $(round(l2_L2_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL2, total_time_rk[rk_L2_fit_start:end], rk_L2_fit, label="RK3 (slope = $(round(rk_L2_slope, sigdigits=3)))", linewidth=linewidth)

scatter!(axL∞, total_time_l0, l0_L∞_errs, markersize=markersize, marker=:circle)
scatter!(axL∞, total_time_l1, l1_L∞_errs, markersize=markersize, marker=:rect)
scatter!(axL∞, total_time_l2, l2_L∞_errs, markersize=markersize, marker=:utriangle)
scatter!(axL∞, total_time_rk, rk_L∞_errs, markersize=markersize, marker=:cross)

lines!(axL∞, total_time_l0[l0_L∞_fit_start:end], l0_L∞_fit, label="FPJ0 (slope = $(round(l0_L∞_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL∞, total_time_l1[l1_L∞_fit_start:end], l1_L∞_fit, label="FPJ1 (slope = $(round(l1_L∞_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL∞, total_time_l2[l2_L∞_fit_start:end], l2_L∞_fit, label="FPJ2 (slope = $(round(l2_L∞_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL∞, total_time_rk[rk_L∞_fit_start:end], rk_L∞_fit, label="RK3 (slope = $(round(rk_L∞_slope, sigdigits=3)))", linewidth=linewidth)

axislegend(axL2, position=:rt)
axislegend(axL∞, position=:lb)
display(fig)
save(joinpath(OUTPUT_DIR, "$(filename_prefix)_combined_error_vs_runtime.png"), fig, px_per_unit=4)
save(joinpath(OUTPUT_DIR, "$(filename_prefix)_combined_error_vs_runtime.pdf"), fig)
