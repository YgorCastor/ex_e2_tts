defmodule ExE2Tts.Model.ODE do
  @moduledoc """
  Implements ODE solvers for trajectory generation in E2/F5 TTS.
  Provides numerical integration methods for solving ordinary differential equations
  used in the flow matching process.
  """

  import Nx.Defn

  @doc """
  Integrates ODE using specified solver method and conditions.

  ## Parameters
    * fn_t - The function representing the ODE (velocity field)
    * y0 - Initial condition
    * ts - Time points for integration
    * opts - Options including:
      * :method - Integration method (:euler | :midpoint), defaults to :euler
      * :rtol - Relative tolerance (not used in fixed-step methods)
      * :atol - Absolute tolerance (not used in fixed-step methods)

  ## Returns
    * Tensor containing the trajectory through time

  ## Examples
      iex> ts = Nx.linspace(0, 1, 32)
      iex> y0 = Nx.tensor([1.0, 0.0])
      iex> fn_t = fn t, y -> -y end
      iex> ODE.odeint(fn_t, y0, ts, method: :midpoint)
  """
  defn odeint(fn_t, y0, ts, opts \\ []) do
    method = Keyword.get(opts, :method, :euler)

    # Input validation
    validate_inputs!(y0, ts)

    # Get timesteps and deltas
    dt = compute_timesteps(ts)

    # Initialize trajectory storage
    trajectory = initialize_trajectory(ts, y0)

    # Integrate ODE
    {_, final_trajectory} =
      while {idx = 0, traj = trajectory}, idx < Nx.axis_size(ts, 0) - 1 do
        t = Nx.take(ts, idx)
        dt_i = Nx.take(dt, idx)
        y = Nx.take(traj, idx)

        # Take integration step based on method
        y_next = integrate_step(method, fn_t, y, t, dt_i)

        # Store result in trajectory
        {idx + 1, Nx.indexed_put(traj, [idx + 1], y_next)}
      end

    final_trajectory
  end

  defnp validate_inputs!(_y0, ts) do
    # Ensure time points are 1D and sorted
    is_valid_ts =
      Nx.all(
        Nx.greater_equal(
          Nx.slice_along_axis(ts, 1, Nx.axis_size(ts, 0) - 1) -
            Nx.slice_along_axis(ts, 0, Nx.axis_size(ts, 0) - 1),
          0.0
        )
      )

    # Assert conditions (will raise during JIT compilation if invalid)
    if Nx.size(ts) < 2 do
      raise ArgumentError, "ts must contain at least 2 time points"
    end

    if not is_valid_ts do
      raise ArgumentError, "ts must be monotonically increasing"
    end
  end

  defnp compute_timesteps(ts) do
    Nx.slice_along_axis(ts, 1, Nx.axis_size(ts, 0) - 1) -
      Nx.slice_along_axis(ts, 0, Nx.axis_size(ts, 0) - 1)
  end

  defnp initialize_trajectory(ts, y0) do
    trajectory = Nx.broadcast(0.0, {Nx.axis_size(ts, 0), Nx.size(y0)})
    Nx.indexed_put(trajectory, [0], y0)
  end

  defnp integrate_step(method, fn_t, y, t, dt) do
    case method do
      :euler -> euler_step(fn_t, y, t, dt)
      :midpoint -> midpoint_step(fn_t, y, t, dt)
    end
  end

  # Performs an Euler integration step.
  # dy/dt = f(t,y) → y_{n+1} = y_n + dt * f(t_n, y_n)
  defnp euler_step(fn_t, y, t, dt) do
    y + dt * fn_t.(t, y)
  end

  # Performs a midpoint integration step.
  # k1 = f(t_n, y_n)
  # k2 = f(t_n + dt/2, y_n + dt/2 * k1)
  # y_{n+1} = y_n + dt * k2
  defnp midpoint_step(fn_t, y, t, dt) do
    k1 = fn_t.(t, y)
    k2 = fn_t.(t + dt / 2, y + dt / 2 * k1)
    y + dt * k2
  end
end
