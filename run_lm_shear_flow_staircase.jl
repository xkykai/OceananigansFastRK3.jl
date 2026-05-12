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
using Dates
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

cfls = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.25, 0.1]
Δts = cfls .* Δx ./ 1.7

cfl_ref = 0.05
Δt_ref = cfl_ref * Δx / 1.7
# Δt_ref = Δts[end] / 4

stop_time = 10

# These are the PR #5059 names for RK3 plus FPJ-0, FPJ-1, and FPJ-2.
timesteppers = [:RungeKutta3, :ConstantPressureProjectionRungeKutta3, :LinearPressureProjectionRungeKutta3, :MidpointPressureProjectionRungeKutta3]

const DATA_DIR = joinpath(@__DIR__, "data")
const OUTPUT_DIR = joinpath(@__DIR__, "output")
mkpath(DATA_DIR)
mkpath(OUTPUT_DIR)

function run_simulation(; timestepper, Δt, filename)
    grid = setup_grid(Nx, Ny, Nz)

    model = setup_model(grid, timestepper)

    simulation = Simulation(model; Δt, stop_time, minimum_relative_step = 1e-10)

    Nt = round(Int, stop_time / Δt)

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
                                                  overwrite_existing = false)

    run!(simulation)
end

#%%
for ts in timesteppers, Δt in Δts
    filename = "$(filename_prefix)_$(ts)_Nx$(Nx)_dt$(Δt).jld2"
    @info "Running $ts with Δt = $Δt"
    run_simulation(timestepper=ts, Δt=Δt, filename=filename)
end

filename_ref = "$(filename_prefix)_RungeKutta3_Nx$(Nx)_dt$(Δt_ref).jld2"
@info "Running REFERENCE RungeKutta3 with Δt = $Δt_ref"
# The reference is used only for final-time error norms in the plotting script.
run_simulation(timestepper=:RungeKutta3, Δt=Δt_ref, filename=filename_ref)

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

# Save y-averaged fields for the manuscript visualization panels.
outputs = (; u=ubar, w=wbar, p=pbar, d=dbar)

filename = joinpath(DATA_DIR, "$(filename_prefix)_RungeKutta3.jld2")
simulation.output_writers[:jld2] = JLD2Writer(model, outputs;
                                            filename = filename,
                                            schedule = TimeInterval(0.1),
                                            overwrite_existing = false)

run!(simulation)
#%%
ntrials = 200

# In the immersed-boundary case, per-step cost depends on Δt through CG behavior.
wall_time_l0 = zeros(length(Δts), ntrials)
wall_time_l1 = zeros(length(Δts), ntrials)
wall_time_l2 = zeros(length(Δts), ntrials)
wall_time_rk = zeros(length(Δts), ntrials)

for (n, Δt) in enumerate(Δts)
    @info "Timing with Δt = $Δt"
    model_l0 = setup_model(grid, :ConstantPressureProjectionRungeKutta3)
    model_l1 = setup_model(grid, :LinearPressureProjectionRungeKutta3)
    model_l2 = setup_model(grid, :MidpointPressureProjectionRungeKutta3)
    model_rk = setup_model(grid, :RungeKutta3)

    for i in 1:ntrials
        t_l0 = @timed time_step!(model_l0, Δt)
        t_l1 = @timed time_step!(model_l1, Δt)
        t_l2 = @timed time_step!(model_l2, Δt)
        t_rk = @timed time_step!(model_rk, Δt)
        wall_time_l0[n, i] = t_l0.time
        wall_time_l1[n, i] = t_l1.time
        wall_time_l2[n, i] = t_l2.time
        wall_time_rk[n, i] = t_rk.time
    end
end

timings_filename = joinpath(DATA_DIR, "$(filename_prefix)_timings.jld2")
isfile(timings_filename) && error("Refusing to overwrite existing timing file: $(timings_filename)")

jldopen(timings_filename, "w") do file
    file["l0"] = wall_time_l0
    file["l1"] = wall_time_l1
    file["l2"] = wall_time_l2
    file["rk"] = wall_time_rk
end




