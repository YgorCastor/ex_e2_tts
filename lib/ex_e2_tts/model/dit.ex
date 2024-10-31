defmodule ExE2Tts.Model.DiT do
  @moduledoc """
  Implements the DiT (Diffusion Transformer) architecture for the E2/F5 TTS model.
  This is a port of the original Python implementation using Axon and Nx.
  """

  import Axon

  alias ExE2Tts.Model.{Attention, FeedForward}

  @doc """
  Creates a new DiT model with the specified configuration.

  ## Parameters
    * opts - Configuration options:
      * dim - Model dimension (default: 1024)
      * depth - Number of transformer layers (default: 22)
      * heads - Number of attention heads (default: 16) 
      * ff_mult - Feed-forward multiplication factor (default: 2)
      * text_dim - Text embedding dimension (default: 512)
      * conv_layers - Number of conv layers for text (default: 4)
      * mel_dim - Mel spectrogram dimension (default: 100)
      * vocab_size - Size of the vocabulary (required)
      * long_skip_connection - Whether to use long skip connections (default: false)
  """
  def create(opts \\ []) do
    # Get base configuration
    config = ExE2Tts.Config.model_config(:f5_tts_base)

    # Merge with default options
    opts =
      Keyword.merge(
        [
          dim: config.dim,
          depth: config.depth,
          heads: config.heads,
          ff_mult: config.ff_mult,
          text_dim: config.text_dim,
          conv_layers: config.conv_layers,
          mel_dim: ExE2Tts.Config.n_mel_channels(),
          long_skip_connection: false,
          vocab_size: opts[:vocab_size] || raise(ArgumentError, "vocab_size is required")
        ],
        opts
      )

    # Create input layers
    input_mel = input("mel", shape: {nil, nil, opts[:mel_dim]})
    input_condition = input("condition", shape: {nil, nil, opts[:mel_dim]})
    input_text = input("text", shape: {nil, nil}, type: :s64)
    input_time = input("time", shape: {nil})
    input_drop_audio = input("drop_audio", shape: {}, type: :u8)
    input_drop_text = input("drop_text", shape: {}, type: :u8)

    # Time embedding
    time_embed = create_time_embedding(input_time, opts[:dim])

    # Text embedding
    text_embed =
      create_text_embedding(input_text,
        vocab_size: opts[:vocab_size],
        dim: opts[:text_dim],
        conv_layers: opts[:conv_layers]
      )

    # Input embedding combining mel, condition and text
    input_embed =
      create_input_embedding(
        input_mel,
        input_condition,
        text_embed,
        input_drop_audio,
        out_dim: opts[:dim]
      )

    # Save input for long skip if enabled
    long_skip = if opts[:long_skip_connection], do: input_embed

    # Build transformer blocks
    transformer_output =
      Enum.reduce(1..opts[:depth], input_embed, fn _i, acc ->
        create_transformer_block(acc,
          dim: opts[:dim],
          heads: opts[:heads],
          dim_head: div(opts[:dim], opts[:heads]),
          ff_mult: opts[:ff_mult],
          dropout: 0.1,
          time_cond: time_embed
        )
      end)

    # Add long skip connection if enabled
    output =
      if opts[:long_skip_connection] do
        transformer_output
        |> concatenate(long_skip)
        |> dense(opts[:dim], name: "long_skip_proj")
      else
        transformer_output
      end

    # Final layer norm and projection
    output =
      output
      |> layer_norm(epsilon: 1.0e-6, name: "final_norm")
      |> dense(opts[:mel_dim], name: "output_proj")

    # Build the model with all inputs
    build(
      {input_mel, input_condition, input_text, input_time, input_drop_audio, input_drop_text},
      output
    )
  end

  # Create time step embedding
  defp create_time_embedding(input, dim) do
    input
    |> sinusoidal_embedding(dim, scale: 1000)
    |> dense(dim, name: "time_proj1")
    |> activation(:silu)
    |> dense(dim, name: "time_proj2")
  end

  # Create text embedding with optional conv layers
  defp create_text_embedding(input, opts) do
    embedded = embedding(input, opts[:vocab_size], opts[:dim], name: "text_embed")

    if opts[:conv_layers] > 0 do
      # Add positional embedding
      embedded = add_positional_embedding(embedded, opts[:dim])

      # Add conv blocks
      Enum.reduce(1..opts[:conv_layers], embedded, fn i, acc ->
        create_conv_block(acc,
          dim: opts[:dim],
          name: "text_conv_#{i}"
        )
      end)
    else
      embedded
    end
  end

  defp create_input_embedding(mel_input, condition_input, text_embed, drop_audio, opts) do
    layer(
      fn {mel, condition, text, drop}, _opts ->
        # Zero out condition if drop_audio is true
        condition =
          Nx.select(
            drop,
            Nx.broadcast(0.0, Nx.shape(condition)),
            condition
          )

        # Concatenate along feature dimension
        Nx.concatenate([mel, condition, text], axis: -1)
      end,
      [{mel_input, condition_input, text_embed, drop_audio}]
    )
    |> dense(opts[:out_dim], name: "input_proj")
    |> create_position_embed(dim: opts[:out_dim])
  end

  defp create_position_embed(input, opts) do
    dim = opts[:dim]
    kernel_size = opts[:kernel_size] || 31
    groups = opts[:groups] || 16

    padding = div(kernel_size - 1, 2)

    input
    |> conv(dim,
      kernel_size: kernel_size,
      padding: padding,
      feature_group_size: groups,
      name: "pos_embed_conv1"
    )
    |> activation(:mish)
    |> conv(dim,
      kernel_size: kernel_size,
      padding: padding,
      feature_group_size: groups,
      name: "pos_embed_conv2"
    )
    |> activation(:mish)
    |> add(input)
  end

  defp create_conv_block(input, opts) do
    dim = opts[:dim]
    kernel_size = opts[:kernel_size] || 7
    dilation = opts[:dilation] || 1

    padding = div(dilation * (kernel_size - 1), 2)

    input
    |> conv(dim,
      kernel_size: kernel_size,
      padding: padding,
      kernel_dilation: dilation,
      feature_group_size: dim,
      name: "#{opts[:name]}_depthwise"
    )
    |> layer_norm(epsilon: 1.0e-6)
    |> activation(:gelu)
    |> conv(dim,
      kernel_size: 1,
      name: "#{opts[:name]}_pointwise"
    )
  end

  # Create a transformer block with AdaLN modulation
  defp create_transformer_block(input, opts) do
    # Pre-norm and modulation for attention
    normed_input =
      ada_layer_norm(input,
        time_cond: opts[:time_cond],
        dim: opts[:dim]
      )

    # Self-attention
    attn_output =
      Attention.create(normed_input,
        dim: opts[:dim],
        heads: opts[:heads],
        dim_head: opts[:dim_head],
        dropout: opts[:dropout]
      )

    # First residual connection
    post_attn = add([input, scale_and_shift(attn_output, opts[:gate_msa])])

    # Pre-norm for feed-forward
    normed_ff = layer_norm(post_attn, epsilon: 1.0e-6)

    # Feed-forward network
    ff_output =
      FeedForward.create(normed_ff,
        dim: opts[:dim],
        mult: opts[:ff_mult],
        dropout: opts[:dropout]
      )

    # Second residual connection
    add([post_attn, scale_and_shift(ff_output, opts[:gate_mlp])])
  end

  # Helper for sinusoidal positional embedding
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

  # Adaptive Layer Normalization
  defp ada_layer_norm(input, opts) do
    time_cond = opts[:time_cond]
    dim = opts[:dim]

    # Project time condition
    {shift, scale, gate} =
      layer(
        fn cond, _opts ->
          hidden = activation(dense(cond, dim), :silu)
          out = dense(hidden, dim * 3)

          # Split into shift, scale and gate
          shift = Nx.slice_along_axis(out, 0, dim)
          scale = Nx.slice_along_axis(out, dim, dim)
          gate = Nx.slice_along_axis(out, 2 * dim, dim)

          {shift, scale, gate}
        end,
        [time_cond]
      )

    # Apply normalization and modulation
    input
    |> layer_norm(epsilon: 1.0e-6)
    |> scale_and_shift(scale, shift)
    |> multiply(gate)
  end

  # Helper for scale and shift operations
  defp scale_and_shift(input, scale, shift \\ nil) do
    if shift do
      add(multiply(input, add(scale, 1.0)), shift)
    else
      multiply(input, add(scale, 1.0))
    end
  end

  # Add positional embedding for text
  defp add_positional_embedding(input, dim) do
    layer(
      fn x, _opts ->
        # Generate position indices
        {_batch, seq_len, _} = Nx.shape(x)
        positions = Nx.iota({seq_len})

        # Create embedding table
        embed_table = create_positional_embedding_table(seq_len, dim)

        # Lookup embeddings
        embeddings = Nx.take(embed_table, positions)

        # Add to input
        x + embeddings
      end,
      [input]
    )
  end

  # Create positional embedding lookup table
  defp create_positional_embedding_table(max_len, dim) do
    positions = Nx.iota({max_len})
    dimensions = Nx.iota({dim})

    angle_rates = Nx.exp(dimensions * -:math.log(10_000) / dim)

    angles = Nx.outer(positions, angle_rates)

    # Interleave sin and cos
    Nx.concatenate(
      [
        Nx.sin(angles),
        Nx.cos(angles)
      ],
      axis: -1
    )
  end
end
