defmodule ExE2Tts.Model.Transformer do
  @moduledoc """
  Implements a transformer block using Axon's high-level API.
  """

  import Axon

  alias ExE2Tts.Model.{Attention, FeedForward}

  @doc """
  Creates a transformer block with attention and feed-forward layers.
  """
  def create(input, opts \\ []) do
    dim = opts[:dim] || 512
    heads = opts[:heads] || 8
    dim_head = opts[:dim_head] || 64
    ff_mult = opts[:ff_mult] || 4
    dropout_rate = opts[:dropout] || 0.0

    # Pre-norm architecture with residual connections
    normed_input =
      layer_norm(input,
        epsilon: 1.0e-6,
        name: "attn_norm"
      )

    # Attention branch with residual
    attention_output =
      normed_input
      |> Attention.create(
        dim: dim,
        heads: heads,
        dim_head: dim_head,
        dropout: dropout_rate
      )

    post_attention =
      add([input, attention_output],
        name: "attention_residual"
      )

    # Feed-forward branch with residual
    normed_ff =
      layer_norm(post_attention,
        epsilon: 1.0e-6,
        name: "ff_norm"
      )

    ff_output =
      normed_ff
      |> FeedForward.create(
        dim: dim,
        mult: ff_mult,
        dropout: dropout_rate
      )

    add([post_attention, ff_output],
      name: "ff_residual"
    )
  end
end
