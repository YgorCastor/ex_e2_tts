defmodule ExE2Tts.Schedulers do
  @moduledoc """
  Schedulers for learning rate and other hyperparameters.
  """
  import Nx.Defn

  def warmup_cosine_decay(init_value, opts \\ []) do
    &apply_warmup_cosine_decay(&1, [{:init_value, init_value} | opts])
  end

  defnp apply_warmup_cosine_decay(step, opts \\ []) do
    opts =
      keyword!(opts,
        init_value: nil,
        warmup_steps: 10,
        decay_steps: 10,
        alpha: 0.0
      )

    init_value = opts[:init_value]
    warmup_steps = opts[:warmup_steps]
    decay_steps = opts[:decay_steps]
    alpha = opts[:alpha]

    # Linear warmup phase
    warmup_rate = init_value / warmup_steps
    warmup = warmup_rate * Nx.min(step, warmup_steps)

    # Cosine decay phase
    decay_step = Nx.max(step - warmup_steps, 0)
    theta = Nx.min(decay_step, decay_steps) / decay_steps * Nx.Constants.pi()
    cos_decay = (Nx.cos(theta) + 1) / 2
    decay = init_value * (cos_decay * (1 - alpha) + alpha)

    # Choose between warmup and decay based on step
    Nx.select(
      Nx.less(step, warmup_steps),
      warmup,
      decay
    )
  end
end
