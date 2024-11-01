defmodule ExE2Tts.Trainer do
  @moduledoc """
  Training loop implementation for E2/F5 TTS.
  """

  require Logger

  @doc """
  Trains the model using Axon's training loop.
  """
  def train(train_dataset, opts \\ []) do
    # Merge with default options
    opts = Keyword.merge(default_options(), opts)
    model = build_model(train_dataset, opts) |> print_model_graph()

    # Setup data loader with dynamic batch sampling
    train_loader =
      ExE2Tts.Dataset.dataloader(train_dataset,
        frames_threshold: opts[:batch_size],
        max_samples: opts[:max_samples],
        random_seed: opts[:resumable_with_seed]
      )

    loss_fn = fn {loss, _cond, _pred}, _target, _opts ->
      # The loss is already computed in CFM.forward as MSE between 
      # predicted flow and actual flow in the masked region
      loss
    end

    # Setup optimizer with warmup and decay
    optimizer = create_optimizer(opts)

    # Setup training loop with Axon
    model
    |> Axon.Loop.trainer(loss_fn, optimizer)
    |> add_ema_handlers(opts[:ema_decay])
    |> Axon.Loop.metric(:loss, "Loss")
    |> Axon.Loop.metric(:learning_rate, "Learning Rate", fn _, state ->
      state.optimizer_state.learning_rate
    end)
    |> add_checkpoint_hooks(opts)
    |> add_logging_hooks(opts)
    |> add_early_stop(opts)
    |> Axon.Loop.run(train_loader, %{ema_model_state: nil},
      epochs: opts[:epochs],
      compiler: EXLA
    )
  end

  defp build_model(dataset, opts) do
    vocab_size = dataset.vocab_size

    case opts[:model_type] do
      :f5_tts ->
        ExE2Tts.Model.DiT.create(
          dim: 1024,
          depth: 22,
          heads: 16,
          ff_mult: 2,
          text_dim: 512,
          conv_layers: 4,
          vocab_size: vocab_size
        )

      :e2_tts ->
        ExE2Tts.Model.UNetT.create(
          dim: 1024,
          depth: 24,
          heads: 16,
          ff_mult: 4,
          vocab_size: vocab_size
        )
    end
  end

  defp create_optimizer(opts) do
    # Create optimizer with warmup and cosine decay
    warmup_steps = opts[:num_warmup_updates]
    total_steps = opts[:epochs] * opts[:steps_per_epoch]
    decay_steps = total_steps - warmup_steps
    initial_value = opts[:learning_rate] * 1.0e-8
    grad_accumulation_steps = opts[:grad_accumulation_steps]
    max_grad_norm = opts[:max_grad_norm]

    Polaris.Optimizers.adam(
      learning_rate:
        ExE2Tts.Schedulers.warmup_cosine_decay(initial_value,
          decay_steps: decay_steps,
          warmup_steps: warmup_steps
        )
    )
    |> Polaris.Updates.accumulate_gradients(grad_accumulation_steps)
    |> Polaris.Updates.clip_by_global_norm(max_grad_norm)
  end

  defp add_checkpoint_hooks(loop, opts) do
    checkpoint_dir = Path.join(opts[:checkpoint_path], "model_{epoch}_{step}.ckpt")
    last_checkpoint_dir = Path.join(opts[:checkpoint_path], "model_last.ckpt")

    loop
    |> Axon.Loop.checkpoint(%{
      filter: [every: opts[:save_per_updates]],
      path: checkpoint_dir
    })
    |> Axon.Loop.checkpoint(%{
      filter: [every: opts[:last_per_steps]],
      path: last_checkpoint_dir
    })
  end

  defp add_logging_hooks(loop, opts) do
    case opts[:logger] do
      :console ->
        loop
        |> Axon.Loop.log(:loss, every: opts[:log_interval])
        |> Axon.Loop.log(:learning_rate, every: opts[:log_interval])

      _ ->
        loop
    end
  end

  defp add_ema_handlers(loop, decay) do
    loop
    |> Axon.Loop.handle_event(:iteration_completed, &update_ema_state(&1, decay))
    |> Axon.Loop.handle_event(:epoch_completed, &log_ema_state/1)
  end

  defp update_ema_state(state, decay) do
    current_model_state = state.model_state
    ema_model_state = state.ema_model_state

    # Initialize EMA state if nil
    ema_model_state =
      if is_nil(ema_model_state) do
        current_model_state
      else
        update_ema_params(ema_model_state, current_model_state, decay)
      end

    %{state | ema_model_state: ema_model_state}
  end

  defp log_ema_state(state) do
    Logger.debug("EMA State updated at epoch #{state.epoch}")
    state
  end

  # EMA formula: decay * ema_value + (1 - decay) * value
  defp update_ema_params(ema_state, current_state, decay) do
    Enum.map(current_state, fn {key, value} ->
      ema_value = ema_state[key]

      updated_value =
        Nx.add(
          Nx.multiply(decay, ema_value),
          Nx.multiply(1 - decay, value)
        )

      {key, updated_value}
    end)
    |> Map.new()
  end

  defp add_early_stop(loop, opts) do
    loop
    |> Axon.Loop.early_stop(
      metric: :loss,
      mode: :min,
      min_delta: opts[:early_stop_min_delta],
      patience: opts[:early_stop_patience]
    )
  end

  defp print_model_graph(model) do
    input_shapes = %{
      # batch, time, mel_channels
      "mel" => {1, 100, 100},
      # batch, sequence_length
      "text" => {1, 50}
    }

    Logger.info("\nModel Graph:")
    Logger.info("=" |> String.duplicate(80))

    try do
      graph = Axon.Display.as_graph(model, input_shapes, direction: :top_down)
      Logger.info(graph)
      model
    rescue
      e ->
        Logger.warning("Could not generate model graph: #{inspect(e)}")
    end
  end

  defp default_options do
    [
      epochs: 100,
      learning_rate: 1.0e-4,
      num_warmup_updates: 2000,
      save_per_updates: 1000,
      checkpoint_path: "checkpoints",
      batch_size: 32,
      batch_size_type: "frame",
      max_samples: 64,
      grad_accumulation_steps: 1,
      max_grad_norm: 1.0,
      logger: :console,
      log_samples: false,
      last_per_steps: 5000,
      log_interval: 100,

      # Model options
      model_type: :f5_tts,
      model_dim: 1024,
      model_depth: 22,
      model_heads: 16,
      model_ff_mult: 2,
      model_text_dim: 512,
      model_conv_layers: 4,

      # Early stopping
      early_stop_min_delta: 1.0e-4,
      early_stop_patience: 10,

      # Compute steps_per_epoch based on dataset
      steps_per_epoch: nil
    ]
  end
end
