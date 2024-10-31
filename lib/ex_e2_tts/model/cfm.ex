defmodule ExE2Tts.Model.CFM do
  @moduledoc """
  Implements Conditional Flow Matching for E2/F5 TTS.
  """

  import Axon
  import ExE2Tts.TextUtils
  import Nx.Defn

  alias ExE2Tts.MelSpec

  defstruct [
    :transformer,
    :mel_spec,
    :odeint_kwargs,
    :vocab_char_map,
    :sigma,
    :audio_drop_prob,
    :cond_drop_prob,
    :num_channels
  ]

  @doc """
  Creates a new CFM instance with specified options.
  """
  def new(transformer, opts \\ []) do
    mel_spec_opts = opts[:mel_spec_kwargs] || %{}
    mel_spec = MelSpec.new(mel_spec_opts)

    %__MODULE__{
      transformer: transformer,
      mel_spec: mel_spec,
      sigma: opts[:sigma] || 0.0,
      odeint_kwargs: opts[:odeint_kwargs] || [method: :euler],
      vocab_char_map: opts[:vocab_char_map],
      audio_drop_prob: opts[:audio_drop_prob] || 0.3,
      cond_drop_prob: opts[:cond_drop_prob] || 0.2,
      num_channels: mel_spec.n_mel_channels
    }
  end

  @doc """
  Builds the main CFM model using Axon.
  """
  def build(opts \\ []) do
    # Input layers matching Python tensor shapes
    # [batch, seq, channels]
    inp = input("input", shape: {nil, nil, nil})
    # [batch, seq]
    text = input("text", shape: {nil, nil}, type: :s64)
    # [batch]
    time = input("time", shape: {nil})
    # [batch, seq, channels]
    cond_input = input("condition", shape: {nil, nil, nil})

    # Create model transforming inputs to outputs
    model =
      {inp, text, time, cond_input}
      |> build_cfm_model(opts)
      # Final projection to mel dimension
      |> dense(opts[:mel_dim] || 100)

    model
  end

  @doc """
  Implements the forward pass for training with flow matching objective.
  """
  defn forward(model, inp, text, opts \\ []) do
    # Process input and lengths
    {x1, lens} = process_input(inp, model.mel_spec)
    batch_size = Nx.axis_size(x1, 0)

    # Add noise to x1 if sigma > 0 
    x1 =
      if model.sigma > 0 do
        noise = Nx.Random.normal(Nx.shape(x1)) * model.sigma
        x1 + noise
      else
        x1
      end

    # Sample noise and time
    x0 = Nx.Random.normal(Nx.shape(x1))
    time = Nx.Random.uniform(shape: {batch_size})

    # Compute flow trajectory
    t = time |> Nx.reshape({:auto, 1, 1})
    phi = (1 - t) * x0 + t * x1
    flow = x1 - x0

    # Create masking for conditional generation
    mask = generate_random_mask(x1, lens, opts)

    # Create conditional input with masking
    conditional = Nx.select(mask, Nx.broadcast(0.0, Nx.shape(x1)), x1)

    # Sample drop probabilities for classifier-free guidance
    {drop_audio, drop_text} = sample_drop_probs(model)

    # Get velocity prediction from transformer
    pred =
      predict_velocity(
        model.transformer,
        phi,
        conditional,
        text,
        time,
        drop_audio_cond: drop_audio,
        drop_text: drop_text,
        opts: opts
      )

    # Calculate loss on masked region
    loss = compute_masked_loss(pred, flow, mask)

    {loss, conditional, pred}
  end

  @doc """
  Implements sampling with ODE solver and classifier-free guidance.
  """
  defn sample(model, cond, text, duration, opts \\ []) do
    # Parse sampling options
    steps = Keyword.get(opts, :steps, 32)
    cfg_strength = Keyword.get(opts, :cfg_strength, 1.0)
    seed = opts[:seed]
    edit_mask = opts[:edit_mask]
    lens = opts[:lens]

    # Validate inputs and compute shapes
    {batch_size, cond_seq_len} = {Nx.axis_size(cond, 0), Nx.axis_size(cond, 1)}
    lens = default_lens(lens, batch_size, cond_seq_len, device(cond))

    # Process text input
    {text, lens} = process_text_input(text, model.vocab_char_map, lens)

    # Create masks and padding
    cond_mask = create_cond_mask(lens, edit_mask)
    cond = pad_conditional(cond, duration, cond_seq_len)

    # Create feature mask for batches > 1
    mask = if batch_size > 1, do: lens_to_mask(duration), else: nil

    # Apply mask to conditional input
    step_cond = Nx.select(cond_mask, cond, Nx.broadcast(0.0, Nx.shape(cond)))

    # Initialize from noise
    y0 = initialize_noise(duration, model.num_channels, batch_size, seed)

    # Create velocity function for ODE solver
    v_fn = fn t, x ->
      pred = predict_velocity(model.transformer, x, step_cond, text, t, mask: mask)

      if cfg_strength > 1.0e-5 do
        # Add classifier-free guidance
        null_pred =
          predict_velocity(
            model.transformer,
            x,
            step_cond,
            text,
            t,
            drop_audio_cond: true,
            drop_text: true,
            mask: mask
          )

        pred + (pred - null_pred) * cfg_strength
      else
        pred
      end
    end

    # Generate trajectory with ODE solver
    ts = Nx.linspace(0, 1, steps)
    {final, trajectory} = solve_ode(v_fn, y0, ts, model.odeint_kwargs)

    # Select final output based on mask
    output = Nx.select(cond_mask, cond, final)

    {output, trajectory}
  end

  defp build_cfm_model({inp, text, time, cond_input}, opts) do
    # Get model configuration
    dim = opts[:dim] || 1024
    num_channels = opts[:mel_dim] || 100

    # Time embedding
    time_embed =
      time
      |> sinusoidal_embedding(dim, scale: 1000)
      |> dense(dim)
      |> activation(:silu)
      |> dense(dim)

    # Text embedding with optional convolutional layers
    text_embed =
      text
      |> ExE2Tts.Model.TextEmbedding.create(
        vocab_size: opts[:vocab_size],
        dim: opts[:text_dim] || dim,
        conv_layers: opts[:conv_layers] || 0
      )

    # Process input features
    processed_input =
      build_input_features(
        inp,
        cond_input,
        text_embed,
        num_channels: num_channels,
        dim: dim
      )

    # Build transformer blocks with time conditioning
    transformer_out =
      build_transformer_blocks(
        processed_input,
        time_embed,
        opts
      )

    # Final normalization and conditioning
    transformer_out
    |> layer_norm(epsilon: 1.0e-6)
    |> add_time_conditioning(time_embed, dim)
  end

  defp build_input_features(inp, cond_input, text_embed, opts) do
    dim = opts[:dim] || raise ArgumentError, "dim is required"

    # Project inputs to common dimension
    projected_inp = dense(inp, dim, name: "input_projection")
    projected_cond = dense(cond_input, dim, name: "condition_projection")

    # Concatenate features and project to model dimension
    concatenate([projected_inp, projected_cond, text_embed])
    |> dense(dim, name: "combined_projection")
    |> add_positional_embedding(dim)
  end

  defp build_transformer_blocks(input, time_embed, opts) do
    depth = opts[:depth] || 24
    heads = opts[:heads] || 16
    ff_mult = opts[:ff_mult] || 4

    # Build transformer layers
    Enum.reduce(1..depth, input, fn i, acc ->
      ExE2Tts.Model.MMDiTBlock.create(
        acc,
        time_embed,
        dim: opts[:dim],
        heads: heads,
        ff_mult: ff_mult,
        name: "transformer_block_#{i}"
      )
    end)
  end

  defp add_positional_embedding(input, dim) do
    kernel_size = 31
    groups = 16
    padding = div(kernel_size - 1, 2)

    input
    |> conv(dim,
      kernel_size: kernel_size,
      padding: padding,
      groups: groups,
      name: "pos_embed_conv1"
    )
    |> activation(:mish)
    |> conv(dim,
      kernel_size: kernel_size,
      padding: padding,
      groups: groups,
      name: "pos_embed_conv2"
    )
    |> activation(:mish)
    |> add(input)
  end

  defp add_time_conditioning(features, time_embed, dim) do
    # Project time embedding to modulation parameters
    time_scale =
      time_embed
      |> dense(dim, name: "time_scale")
      |> activation(:sigmoid)

    time_shift =
      time_embed
      |> dense(dim, name: "time_shift")

    # Apply conditioning
    features
    |> multiply(add(time_scale, 1.0))
    |> add(time_shift)
  end

  defp sinusoidal_embedding(input, dim, opts) do
    scale = opts[:scale] || 1.0

    layer(
      fn x, _opts ->
        half_dim = div(dim, 2)
        emb = :math.log(10_000) / (half_dim - 1)

        # Generate position encodings
        positions = Nx.reshape(x, {:auto, 1})
        dim_pos = Nx.iota({half_dim})

        angles = positions * Nx.exp(-emb * dim_pos) * scale

        # Generate sin and cos embeddings
        Nx.concatenate(
          [
            Nx.sin(angles),
            Nx.cos(angles)
          ],
          axis: -1
        )
      end,
      [input]
    )
  end

  # Private helpers

  defnp compute_masked_loss(pred, flow, mask) do
    loss = Nx.pow(pred - flow, 2)
    loss = Nx.select(mask, loss, Nx.broadcast(0.0, Nx.shape(loss)))
    Nx.mean(loss)
  end

  defnp generate_random_mask(input, lens, opts) do
    {min_frac, max_frac} = Keyword.get(opts, :frac_range, {0.7, 1.0})

    {batch_size, seq_len, _} = Nx.shape(input)

    frac_lengths =
      Nx.Random.uniform(
        shape: {batch_size},
        min: min_frac,
        max: max_frac
      )

    case lens do
      nil -> mask_from_frac_lengths(seq_len, frac_lengths)
      lens -> generate_mask(lens, batch_size, frac_lengths, seq_len: seq_len)
    end
  end

  defnp predict_velocity(transformer, x, conditional, text, time, opts) do
    transformer.(
      x,
      conditional,
      text,
      time,
      drop_audio_cond: Keyword.get(opts, :drop_audio_cond, false),
      drop_text: Keyword.get(opts, :drop_text, false),
      mask: opts[:mask]
    )
  end

  defnp sample_drop_probs(model) do
    drop_audio = Nx.Random.uniform() < model.audio_drop_prob
    drop_text = if Nx.Random.uniform() < model.cond_drop_prob, do: true, else: drop_audio
    {drop_audio, drop_text}
  end

  defnp solve_ode(v_fn, y0, ts, opts) do
    ExE2Tts.Model.ODE.odeint(v_fn, y0, ts, opts)
  end

  defnp create_cond_mask(lens, edit_mask) do
    mask = lens_to_mask(lens)

    if edit_mask do
      Nx.logical_and(mask, edit_mask)
    else
      mask
    end
    |> Nx.reshape({:auto, :auto, 1})
  end

  defnp pad_conditional(cond, duration, cond_seq_len) do
    max_duration = Nx.reduce_max(duration)
    padding = max_duration - cond_seq_len

    Nx.pad(
      cond,
      [{0, 0}, {0, padding}, {0, 0}],
      value: 0.0
    )
  end

  defnp initialize_noise(duration, channels, batch_size, seed) do
    shape = {batch_size, duration, channels}

    if seed do
      key = Nx.Random.key(seed)
      Nx.Random.normal(key, shape: shape)
    else
      Nx.Random.normal(shape)
    end
  end

  defnp process_input(inp, mel_spec) do
    case Nx.shape(inp) do
      # Raw audio
      {_, _} ->
        mel = ExE2Tts.MelSpec.audio_to_mel(mel_spec, inp)
        mel = Nx.transpose(mel, [0, 2, 1])
        {mel, nil}

      # Already mel spectrogram
      {_, _, _} ->
        lens = Nx.broadcast(Nx.axis_size(inp, 1), {Nx.axis_size(inp, 0)})
        {inp, lens}
    end
  end

  defnp mask_from_frac_lengths(seq_len, frac_lengths) do
    # Calculate actual lengths to mask
    lengths =
      frac_lengths
      |> Nx.multiply(seq_len)
      |> Nx.as_type(:s64)

    # Calculate maximum starting positions
    max_start = Nx.subtract(seq_len, lengths)

    # Generate random start positions
    rand = Nx.Random.uniform(Nx.shape(frac_lengths))

    start =
      Nx.multiply(max_start, rand)
      |> Nx.as_type(:s64)
      |> Nx.clip(0, Nx.subtract(seq_len, 1))

    # Calculate end positions
    end_ = Nx.add(start, lengths)

    # Generate mask from start and end indices
    mask_from_start_end_indices(seq_len, start, end_)
  end

  defnp mask_from_start_end_indices(seq_len, start, end_) do
    # Create sequence indices
    seq = Nx.iota({seq_len})
    # Broadcast sequence for batch processing
    seq = Nx.broadcast(seq, {Nx.axis_size(start, 0), seq_len})

    # Create masks for start and end positions
    start_mask = Nx.greater_equal(seq, Nx.reshape(start, {:auto, 1}))
    end_mask = Nx.less(seq, Nx.reshape(end_, {:auto, 1}))

    # Combine masks
    Nx.logical_and(start_mask, end_mask)
  end

  defnp generate_mask(lens, batch_size, frac_lengths, opts \\ []) do
    seq_len = Keyword.get(opts, :seq_len, Nx.axis_size(lens, 1))

    # Create indices sequence
    seq_indices = Nx.iota({seq_len})

    # Calculate mask boundaries
    {starts, ends} = compute_mask_boundaries(lens, frac_lengths)

    # Reshape for broadcasting
    starts = Nx.reshape(starts, {:auto, 1})
    ends = Nx.reshape(ends, {:auto, 1})

    # Broadcast sequence indices
    seq_broadcast = Nx.broadcast(seq_indices, {batch_size, seq_len})

    # Create mask
    mask =
      Nx.logical_and(
        Nx.greater_equal(seq_broadcast, starts),
        Nx.less(seq_broadcast, ends)
      )

    # Apply length masking if needed
    case lens do
      nil ->
        mask

      _ ->
        length_mask = create_length_mask(lens, seq_len)
        Nx.logical_and(mask, length_mask)
    end
  end

  defnp compute_mask_boundaries(lens, frac_lengths) do
    # Calculate lengths for masked regions
    masked_lengths = Nx.multiply(lens, frac_lengths)

    # Calculate maximum possible start positions
    max_starts = Nx.subtract(lens, masked_lengths)

    # Generate random starting positions
    random_fracs = Nx.Random.uniform(shape: {Nx.axis_size(lens, 0)})
    starts = Nx.multiply(max_starts, random_fracs)
    starts = Nx.clip(starts, 0, Nx.subtract(lens, 1))

    # Calculate end positions
    ends = Nx.add(starts, masked_lengths)
    ends = Nx.clip(ends, 0, lens)

    {starts, ends}
  end

  defnp create_length_mask(lens, seq_len) do
    batch_size = Nx.axis_size(lens, 0)

    # Create sequence position indices
    seq_indices =
      Nx.broadcast(
        Nx.iota({seq_len}),
        {batch_size, seq_len}
      )

    # Broadcast lengths for comparison
    lengths_broadcast = Nx.reshape(lens, {:auto, 1})

    # Create mask where position is less than sequence length
    Nx.less(seq_indices, lengths_broadcast)
  end
end
