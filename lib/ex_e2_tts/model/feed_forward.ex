defmodule ExE2Tts.Model.FeedForward do
  @moduledoc """
  Implements feed-forward network using Axon's high-level API.
  """

  import Axon

  @doc """
  Creates a feed-forward network with expansion and projection.
  """
  def create(input, opts \\ []) do
    dim = opts[:dim] || 512
    mult = opts[:mult] || 4
    dropout_rate = opts[:dropout] || 0.0

    inner_dim = dim * mult

    input
    |> dense(inner_dim,
      kernel_initializer: :glorot_uniform,
      name: "ff_expand"
    )
    |> activation(:gelu)
    |> dropout(rate: dropout_rate)
    |> dense(dim,
      kernel_initializer: :glorot_uniform,
      name: "ff_project"
    )
    |> dropout(rate: dropout_rate)
  end
end
