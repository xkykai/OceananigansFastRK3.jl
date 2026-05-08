using Oceananigans
using Oceananigans.Units
using Printf
using JLD2
using Statistics
# using CairoMakie
using CUDA
using CairoMakie
using BenchmarkTools
using Oceananigans: prognostic_fields
using Oceananigans.TimeSteppers: RungeKutta3TimeStepper, LeMoinRungeKutta3TimeStepper

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

cfls = [0.8, 0.4, 0.3, 0.2, 0.1, 0.05]
Δts = cfls .* Δx ./ 1
Δt_ref = Δts[end] / 8

stop_time = 10

# timestepper = :RungeKutta3
# timestepper = :LeMoinRungeKutta3
# timesteppers = [:RungeKutta3]
timesteppers = [:RungeKutta3, :LeMoinRungeKutta3FPJ0, :LeMoinRungeKutta3FPJ1, :LeMoinRungeKutta3FPJ2]

const DATA_DIR = joinpath(@__DIR__, "data")
mkpath(DATA_DIR)

function run_simulation(; timestpr, Δt, filename)
    grid = setup_grid(Nx, Ny, Nz)

    model_prototype = setup_model(grid)
    
    if timestpr == :LeMoinRungeKutta3FPJ0
        α = β = 0
        timestepper = LeMoinRungeKutta3TimeStepper(grid, prognostic_fields(model_prototype); α, β)
    elseif timestpr == :LeMoinRungeKutta3FPJ1
        α = 1
        β = 0
        timestepper = LeMoinRungeKutta3TimeStepper(grid, prognostic_fields(model_prototype); α, β)
    elseif timestpr == :LeMoinRungeKutta3FPJ2
        α = 1 // 2
        β = 1 // 2
        timestepper = LeMoinRungeKutta3TimeStepper(grid, prognostic_fields(model_prototype); α, β)
    else
        timestepper = RungeKutta3TimeStepper(grid, prognostic_fields(model_prototype))
    end

    model = setup_model(grid, timestepper)
    initial_conditions!(model)

    simulation = Simulation(model; Δt, stop_time, minimum_relative_step = 1e-10)

    Nt = round(Int, stop_time / Δt)

    u, v, w = model.velocities

    d = Field(∂x(u) + ∂y(v) + ∂z(w))
    wall_time = Ref(time_ns())

    function progress(sim)
        msg = @sprintf("Iter: %d, time: %s, Δt: %6.3e",
                        iteration(sim), prettytime(time(sim)), sim.Δt)

        elapsed = 1e-9 * (time_ns() - wall_time[])

        compute!(d)

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
    filename = "$(filename_prefix)_$(ts)_Nx$(Nx)_Δt$(Δt).jld2"
    @info "Running $ts with Δt = $Δt"
    run_simulation(timestpr=ts, Δt=Δt, filename=filename)
end

filename_ref = "$(filename_prefix)_RungeKutta3_Nx$(Nx)_Δt$(Δt_ref).jld2"
@info "Running REFERENCE RungeKutta3 with Δt = $Δt_ref"
run_simulation(timestpr=:RungeKutta3, Δt=Δt_ref, filename=filename_ref)

#%%
rk_filenames = ["$(filename_prefix)_RungeKutta3_Nx$(Nx)_Δt$(Δt).jld2" for Δt in Δts]
l0_filenames = ["$(filename_prefix)_LeMoinRungeKutta3FPJ0_Nx$(Nx)_Δt$(Δt).jld2" for Δt in Δts]
l1_filenames = ["$(filename_prefix)_LeMoinRungeKutta3FPJ1_Nx$(Nx)_Δt$(Δt).jld2" for Δt in Δts]
l2_filenames = ["$(filename_prefix)_LeMoinRungeKutta3FPJ2_Nx$(Nx)_Δt$(Δt).jld2" for Δt in Δts]
ref_filename = "$(filename_prefix)_RungeKutta3_Nx$(Nx)_Δt$(Δt_ref).jld2"

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

u_ref = interior(ref_u_data[length(ref_u_data.times)], :, :, :)
w_ref = interior(ref_w_data[length(ref_w_data.times)], :, :, :)

function final_time_errors(ut, wt)
    n = length(ut.times)

    u_num = interior(ut[n], :, :, :)
    w_num = interior(wt[n], :, :, :)

    eu = u_num .- u_ref
    ew = w_num .- w_ref

    return (; L∞_u = maximum(abs, eu), L∞_w = maximum(abs, ew),
              L2_u = sqrt(mean(eu.^2)), L2_w = sqrt(mean(ew.^2)))
end

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
fig = Figure(size=(900, 500), fontsize=15)
axL2 = Axis(fig[1, 1], title="L2 error at final time", xlabel="Δt", ylabel="L2 error", xscale=log10, yscale=log10)
axL∞ = Axis(fig[1, 2], title="L∞ error at final time", xlabel="Δt", ylabel="L∞ error", xscale=log10, yscale=log10)

lm_fit_start = 3
markersize = 10
linewidth = 3
l0_L2_u_slope, l0_L2_u_fit = fit_power_law(Δts[lm_fit_start:end], l0_L2_u_errs[lm_fit_start:end])
l0_L2_w_slope, l0_L2_w_fit = fit_power_law(Δts[lm_fit_start:end], l0_L2_w_errs[lm_fit_start:end])
l0_L∞_u_slope, l0_L∞_u_fit = fit_power_law(Δts[lm_fit_start:end], l0_L∞_u_errs[lm_fit_start:end])
l0_L∞_w_slope, l0_L∞_w_fit = fit_power_law(Δts[lm_fit_start:end], l0_L∞_w_errs[lm_fit_start:end])
l1_L2_u_slope, l1_L2_u_fit = fit_power_law(Δts[lm_fit_start:end], l1_L2_u_errs[lm_fit_start:end])
l1_L2_w_slope, l1_L2_w_fit = fit_power_law(Δts[lm_fit_start:end], l1_L2_w_errs[lm_fit_start:end])
l1_L∞_u_slope, l1_L∞_u_fit = fit_power_law(Δts[lm_fit_start:end], l1_L∞_u_errs[lm_fit_start:end])
l1_L∞_w_slope, l1_L∞_w_fit = fit_power_law(Δts[lm_fit_start:end], l1_L∞_w_errs[lm_fit_start:end])
l2_L2_u_slope, l2_L2_u_fit = fit_power_law(Δts[lm_fit_start:end], l2_L2_u_errs[lm_fit_start:end])
l2_L2_w_slope, l2_L2_w_fit = fit_power_law(Δts[lm_fit_start:end], l2_L2_w_errs[lm_fit_start:end])
l2_L∞_u_slope, l2_L∞_u_fit = fit_power_law(Δts[lm_fit_start:end], l2_L∞_u_errs[lm_fit_start:end])
l2_L∞_w_slope, l2_L∞_w_fit = fit_power_law(Δts[lm_fit_start:end], l2_L∞_w_errs[lm_fit_start:end])

rk_L2_u_slope, rk_L2_u_fit = fit_power_law(Δts, rk_L2_u_errs)
rk_L2_w_slope, rk_L2_w_fit = fit_power_law(Δts, rk_L2_w_errs)
rk_L∞_u_slope, rk_L∞_u_fit = fit_power_law(Δts, rk_L∞_u_errs)
rk_L∞_w_slope, rk_L∞_w_fit = fit_power_law(Δts, rk_L∞_w_errs)

scatter!(axL2, Δts, l0_L2_u_errs, markersize=markersize)
scatter!(axL2, Δts, l0_L2_w_errs, markersize=markersize)
scatter!(axL2, Δts, l1_L2_u_errs, markersize=markersize)
scatter!(axL2, Δts, l1_L2_w_errs, markersize=markersize)
scatter!(axL2, Δts, l2_L2_u_errs, markersize=markersize)
scatter!(axL2, Δts, l2_L2_w_errs, markersize=markersize)
scatter!(axL2, Δts, rk_L2_u_errs, markersize=markersize)
scatter!(axL2, Δts, rk_L2_w_errs, markersize=markersize)

lines!(axL2, Δts[lm_fit_start:end], l0_L2_u_fit, label="FPJ0-RK3 u (slope = $(round(l0_L2_u_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL2, Δts[lm_fit_start:end], l0_L2_w_fit, label="FPJ0-RK3 w (slope = $(round(l0_L2_w_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL2, Δts[lm_fit_start:end], l1_L2_u_fit, label="FPJ1-RK3 u (slope = $(round(l1_L2_u_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL2, Δts[lm_fit_start:end], l1_L2_w_fit, label="FPJ1-RK3 w (slope = $(round(l1_L2_w_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL2, Δts[lm_fit_start:end], l2_L2_u_fit, label="FPJ2-RK3 u (slope = $(round(l2_L2_u_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL2, Δts[lm_fit_start:end], l2_L2_w_fit, label="FPJ2-RK3 w (slope = $(round(l2_L2_w_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL2, Δts, rk_L2_u_fit, label="RK3 u (slope = $(round(rk_L2_u_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL2, Δts, rk_L2_w_fit, label="RK3 w (slope = $(round(rk_L2_w_slope, sigdigits=3)))", linewidth=linewidth)

scatter!(axL∞, Δts, l0_L∞_u_errs, markersize=markersize)
scatter!(axL∞, Δts, l0_L∞_w_errs, markersize=markersize)
scatter!(axL∞, Δts, l1_L∞_u_errs, markersize=markersize)
scatter!(axL∞, Δts, l1_L∞_w_errs, markersize=markersize)
scatter!(axL∞, Δts, l2_L∞_u_errs, markersize=markersize)
scatter!(axL∞, Δts, l2_L∞_w_errs, markersize=markersize)
scatter!(axL∞, Δts, rk_L∞_u_errs, markersize=markersize)
scatter!(axL∞, Δts, rk_L∞_w_errs, markersize=markersize)
lines!(axL∞, Δts[lm_fit_start:end], l0_L∞_u_fit, label="FPJ0-RK3 u (slope = $(round(l0_L∞_u_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL∞, Δts[lm_fit_start:end], l0_L∞_w_fit, label="FPJ0-RK3 w (slope = $(round(l0_L∞_w_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL∞, Δts[lm_fit_start:end], l1_L∞_u_fit, label="FPJ1-RK3 u (slope = $(round(l1_L∞_u_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL∞, Δts[lm_fit_start:end], l1_L∞_w_fit, label="FPJ1-RK3 w (slope = $(round(l1_L∞_w_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL∞, Δts[lm_fit_start:end], l2_L∞_u_fit, label="FPJ2-RK3 u (slope = $(round(l2_L∞_u_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL∞, Δts[lm_fit_start:end], l2_L∞_w_fit, label="FPJ2-RK3 w (slope = $(round(l2_L∞_w_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL∞, Δts, rk_L∞_u_fit, label="RK3 u (slope = $(round(rk_L∞_u_slope, sigdigits=3)))", linewidth=linewidth)
lines!(axL∞, Δts, rk_L∞_w_fit, label="RK3 w (slope = $(round(rk_L∞_w_slope, sigdigits=3)))", linewidth=linewidth)
axislegend(axL2, position=:lt)
axislegend(axL∞, position=:lt)
display(fig)
save("./$(filename_prefix)_convergence.png", fig, px_per_unit=4)
save("./$(filename_prefix)_convergence.pdf", fig)

#%%
model = setup_model(grid, :RungeKutta3)
Δt = 0.5 * Δx / 10

simulation = Simulation(model; Δt, stop_time, minimum_relative_step = 1e-10)

time_wizard = TimeStepWizard(cfl=0.6, max_change=1.05, min_Δt=1e-5, max_Δt=10)
simulation.callbacks[:wizard] = Callback(time_wizard, IterationInterval(1))

u, v, w = model.velocities

d = Field(∂x(u) + ∂y(v) + ∂z(w))
wall_time = Ref(time_ns())

function progress(sim)
    msg = @sprintf("Iter: %d, time: %s, Δt: %6.3e",
                    iteration(sim), prettytime(time(sim)), sim.Δt)

    elapsed = 1e-9 * (time_ns() - wall_time[])

    compute!(d)

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

CairoMakie.record(fig, "./$(filename_prefix)_RK3.mp4", 1:Nt, framerate=15) do nn
    n[] = nn
end

#%%
model_l0 = setup_model(grid, :LeMoinRungeKutta3FPJ0)
model_l1 = setup_model(grid, :LeMoinRungeKutta3FPJ1)
model_l2 = setup_model(grid, :LeMoinRungeKutta3FPJ2)
model_rk = setup_model(grid, :RungeKutta3)

wall_time_l0 = zeros(200)
wall_time_l1 = zeros(200)
wall_time_l2 = zeros(200)
wall_time_rk = zeros(200)

for i in eachindex(wall_time_rk)
    t_l0 = @timed time_step!(model_l0, Δt_ref)
    t_l1 = @timed time_step!(model_l1, Δt_ref)
    t_l2 = @timed time_step!(model_l2, Δt_ref)
    t_rk = @timed time_step!(model_rk, Δt_ref)
    wall_time_l0[i] = t_l0.time
    wall_time_l1[i] = t_l1.time
    wall_time_l2[i] = t_l2.time
    wall_time_rk[i] = t_rk.time
end

median_wall_time_l0 = median(wall_time_l0)
median_wall_time_l1 = median(wall_time_l1)
median_wall_time_l2 = median(wall_time_l2)
median_wall_time_rk = median(wall_time_rk)

Nts = stop_time ./ Δts

total_time_l0 = median_wall_time_l0 .* Nts
total_time_l1 = median_wall_time_l1 .* Nts
total_time_l2 = median_wall_time_l2 .* Nts
total_time_rk = median_wall_time_rk .* Nts
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
save("./$(filename_prefix)_convergence_wall_time.png", fig, px_per_unit=4)
save("./$(filename_prefix)_convergence_wall_time.pdf", fig)