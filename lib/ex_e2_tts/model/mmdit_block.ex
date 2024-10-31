defmodule ExE2Tts.Model.MMDiTBlock do
  @moduledoc """
  Implements the MMDiT (Modulated Mixed-Data Transformer) Block.
  Matches the transformer block implementation from the Python version with AdaLN.
  """

  import Axon

  alias ExE2Tts.Model.{Attention, FeedForward}

  @doc """
  Creates a new MMDiT block with attention and feed-forward layers.
  Implements adaptive layer normalization and time conditioning.

  ## Parameters
    * input - The input tensor to transform
    * time_embed - Time embedding tensor for conditioning
    * opts - Configuration options:
      * dim - Model dimension (default: 1024)
      * heads - Number of attention heads (default: 16)
      * dim_head - Dimension per head (default: 64)
      * ff_mult - Feed forward expansion factor (default: 4)
      * dropout - Dropout rate (default: 0.1)
      * context_pre_only - Whether this is the final layer (default: false)
  """
  def create(input, time_embed, opts \\ []) do
    opts =
      Keyword.merge(
        [
          dim: 1024,
          heads: 16,
          dim_head: 64,
          ff_mult: 4,
          dropout: 0.1,
          context_pre_only: false
        ],
        opts
      )

    # Pre-norm and modulation for attention
    {norm_input, gate_msa, shift_mlp, scale_mlp, gate_mlp} =
      ada_layer_norm(input, time_embed,
        dim: opts[:dim],
        name: "#{opts[:name]}_norm1"
      )

    # Self-attention with residual
    attn_output =
      norm_input
      |> attention_block(
        dim: opts[:dim],
        heads: opts[:heads],
        dim_head: opts[:dim_head],
        dropout: opts[:dropout],
        name: "#{opts[:name]}_attn"
      )

    # First residual connection
    post_attn =
      add_gated_residual(
        input,
        attn_output,
        gate_msa,
        name: "#{opts[:name]}_attn_residual"
      )

    if opts[:context_pre_only] do
      # Final layer only does pre-norm
      post_attn
    else
      # Pre-norm for feed-forward
      normed_ff = layer_norm(post_attn, epsilon: 1.0e-6)

      # Scale and shift from time conditioning
      normed_ff =
        normed_ff
        |> scale_and_shift(scale_mlp, shift_mlp)

      # Feed-forward network
      ff_output =
        normed_ff
        |> FeedForward.create(
          dim: opts[:dim],
          mult: opts[:ff_mult],
          dropout: opts[:dropout],
          name: "#{opts[:name]}_ff"
        )

      # Second residual connection with gating
      add_gated_residual(
        post_attn,
        ff_output,
        gate_mlp,
        name: "#{opts[:name]}_ff_residual"
      )
    end
  end

  defp attention_block(input, opts) do
    Attention.create(input,
      dim: opts[:dim],
      heads: opts[:heads],
      dim_head: opts[:dim_head],
      dropout: opts[:dropout],
      name: opts[:name]
    )
  end

  defp ada_layer_norm(input, time_emb, opts) do
    # Project time embedding to get modulation parameters (6 components)
    modulation =
      time_emb
      |> dense(opts[:dim] * 6, name: "#{opts[:name]}_mod")
      |> activation(:silu)

    # Split into shift, scale and gates
    {shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp} =
      split_modulation(modulation, opts[:dim])

    # Apply normalization and modulation
    normed =
      input
      |> layer_norm(epsilon: 1.0e-6, name: opts[:name])
      |> scale_and_shift(scale_msa, shift_msa)

    {normed, gate_msa, shift_mlp, scale_mlp, gate_mlp}
  end

  defp split_modulation(modulation, _dim) do
    layer(
      fn x, _opts ->
        # Split along feature dimension into 6 equal parts
        chunks = Nx.split(x, 6, axis: -1)

        case chunks do
          [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp] ->
            {shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp}

          _ ->
            raise "Expected 6 modulation components but got #{length(chunks)}"
        end
      end,
      [modulation]
    )
  end

  defp scale_and_shift(input, scale, shift) do
    input
    |> multiply(add(scale, 1.0))
    |> add(shift)
  end

  defp add_gated_residual(residual, output, gate, opts) do
    gated_output =
      output
      |> multiply(gate)
      |> add(residual)

    # Optional layer normalization
    if opts[:normalize] do
      layer_norm(gated_output, epsilon: 1.0e-6, name: "#{opts[:name]}_out_norm")
    else
      gated_output
    end
  end
end
