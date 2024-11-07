defmodule ExE2Tts.Model.UNetT do
  @moduledoc """
  Implements the UNetT (U-Net Transformer) architecture for E2 TTS.
  This is a flat U-Net style transformer that uses skip connections
  between corresponding layers in the first and second half of the network.
  """

  import Axon

  alias ExE2Tts.Model.{RotaryEmbedding, Transformer, TextEmbedding}

  @doc """
  Creates the complete UNetT model.

  ## Parameters
    * opts - Options for configuring the model
      * dim - Model dimension (default: 1024)
      * depth - Number of transformer layers (must be even) (default: 24)
      * heads - Number of attention heads (default: 16)
      * ff_mult - Feed-forward multiplication factor (default: 4)
      * vocab_size - Size of the vocabulary (required)
      * skip_connect_type - Type of skip connection (:add | :concat | :none) (default: :concat)
  """
  def create(opts \\ []) do
    # Model configuration
    config = ExE2Tts.Config.model_config(:e2_tts_base)

    opts =
      Keyword.merge(
        [
          dim: config.dim,
          depth: config.depth,
          heads: config.heads,
          dim_head: 64,
          ff_mult: config.ff_mult,
          mel_dim: ExE2Tts.Config.n_mel_channels(),
          dropout: 0.1,
          conv_layers: 0,
          skip_connect_type: :concat,
          vocab_size: opts[:vocab_size] || raise(ArgumentError, "vocab_size is required")
        ],
        opts
      )

    # Validate depth is even
    if rem(opts[:depth], 2) != 0 do
      raise ArgumentError, "depth must be even for UNetT"
    end

    # Input layers
    input_mel = input("mel", shape: {nil, nil, opts[:mel_dim]})
    input_text = input("text", shape: {nil, nil}, type: :s64)
    input_time = input("time", shape: {nil})

    # Process inputs with additional features
    mel_features =
      input_mel
      |> process_mel(opts[:dim])
      |> add_conv_layers(opts[:conv_layers], opts[:dim], opts[:dropout])

    text_features =
      TextEmbedding.create(input_text,
        vocab_size: opts[:vocab_size],
        dim: opts[:dim],
        # Pass through conv_layers option
        conv_layers: opts[:conv_layers]
      )

    time_features = process_time(input_time, opts[:dim])

    # Initialize rotary embeddings
    rotary = RotaryEmbedding.new(opts[:dim_head])

    # Combine features and append time token
    combined = combine_features(mel_features, text_features, time_features)

    # Build UNetT architecture
    half_depth = div(opts[:depth], 2)

    # First half of the network (down path)
    {down_features, skips} = build_down_path(combined, half_depth, rotary, opts)

    # Second half of the network (up path with skip connections)
    output = build_up_path(down_features, skips, half_depth, rotary, opts)

    # Final processing
    output =
      output
      |> layer_norm(epsilon: 1.0e-6, name: "final_norm")
      |> dropout(rate: opts[:dropout], name: "final_dropout")
      |> dense(opts[:mel_dim], name: "output_projection")

    # Build model
    Axon.build({input_mel, input_text, input_time}, output)
  end

  defp process_mel(input, dim) do
    input
    |> dense(dim, name: "mel_projection")
    |> layer_norm(epsilon: 1.0e-6, name: "mel_norm")
  end

  defp process_time(input, dim) do
    input
    |> sinusoidal_position_embedding(dim)
    |> dense(dim, name: "time_proj1")
    |> activation(:silu)
    |> dense(dim, name: "time_proj2")
  end

  defp add_conv_layers(input, conv_layers, dim, dropout_rate) when conv_layers > 0 do
    Enum.reduce(1..conv_layers, input, fn i, acc ->
      acc
      |> conv(dim,
        kernel_size: 3,
        padding: :same,
        use_bias: true,
        channels: :last,
        name: "conv_layer_#{i}",
        activation: :mish
      )
      |> batch_norm(name: "conv_bn_#{i}")
      |> dropout(rate: dropout_rate, name: "conv_dropout_#{i}")
    end)
  end

  defp add_conv_layers(input, _conv_layers, _dim, _dropout_rate), do: input

  defp sinusoidal_position_embedding(input, dim) do
    layer(
      fn x, _opts ->
        half_dim = div(dim, 2)
        emb = :math.log(10_000) / (half_dim - 1)

        positions = Nx.reshape(x, {:auto, 1})
        dim_indices = Nx.iota({half_dim})

        angle_rates = Nx.exp(dim_indices * -emb)
        angle_rads = positions * angle_rates

        sin_emb = Nx.sin(angle_rads)
        cos_emb = Nx.cos(angle_rads)

        Nx.concatenate([sin_emb, cos_emb], axis: -1)
      end,
      [input],
      name: "sinusoidal_embedding"
    )
  end

  defp combine_features(mel, text, time) do
    layer(
      fn {mel, text, time}, _opts ->
        {batch, seq_len, _} = Nx.shape(mel)
        time = Nx.reshape(time, {batch, 1, -1})

        # Append time token to combined features
        features = Nx.concatenate([mel, text], axis: -1)

        Nx.broadcast(features, {batch, seq_len, -1})
        |> Nx.add(time)
      end,
      [{mel, text, time}],
      name: "feature_combination"
    )
  end

  defp build_down_path(input, depth, rotary, opts) do
    Enum.reduce(1..depth, {input, []}, fn i, {features, skips} ->
      # Get sequence length for rotary embeddings
      seq_len = elem(Nx.shape(features), 1)

      # Get rotary embeddings for current sequence length
      {freqs, _} = RotaryEmbedding.forward_from_seq_len(rotary, seq_len)

      # Apply transformer block with rotary embeddings
      output =
        Transformer.create(features,
          dim: opts[:dim],
          heads: opts[:heads],
          dim_head: opts[:dim_head],
          dropout: opts[:dropout],
          ff_mult: opts[:ff_mult],
          rotary_embeddings: freqs,
          name: "down_block_#{i}"
        )

      # Store skip connection
      {output, [features | skips]}
    end)
  end

  defp build_up_path(input, skips, depth, rotary, opts) do
    Enum.reduce(1..depth, {input, skips}, fn i, {features, [skip | remaining_skips]} ->
      # Apply skip connection
      features = apply_skip_connection(features, skip, opts[:skip_connect_type])

      # Get sequence length for rotary embeddings
      seq_len = elem(Nx.shape(features), 1)

      # Get rotary embeddings for current sequence length
      {freqs, _} = RotaryEmbedding.forward_from_seq_len(rotary, seq_len)

      # Apply transformer block with rotary embeddings
      output =
        Transformer.create(features,
          dim: opts[:dim],
          heads: opts[:heads],
          dim_head: opts[:dim_head],
          dropout: opts[:dropout],
          ff_mult: opts[:ff_mult],
          rotary_embeddings: freqs,
          name: "up_block_#{i}"
        )

      {output, remaining_skips}
    end)
    |> elem(0)
  end

  defp apply_skip_connection(features, skip, type) do
    case type do
      :concat ->
        # Concatenate and project back to original dimension
        layer(
          fn {x, skip}, _opts ->
            Nx.concatenate([x, skip], axis: -1)
          end,
          [{features, skip}],
          name: "skip_concat"
        )
        |> dense(elem(Nx.shape(features), -1), name: "skip_proj")

      :add ->
        Axon.add([features, skip], name: "skip_add")

      :none ->
        features
    end
  end
end
