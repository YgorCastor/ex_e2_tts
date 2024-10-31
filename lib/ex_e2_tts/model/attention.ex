defmodule ExE2Tts.Model.Attention do
  @moduledoc """
  Implements scaled dot-product attention mechanism.
  """

  import Axon

  @doc """
  Creates a multi-head attention layer with specified dimensions.

  ## Parameters
    * input - The input layer to build upon
    * opts - Options for configuring the attention mechanism
  """
  def create(input, opts \\ []) do
    dim = opts[:dim] || 512
    n_heads = opts[:heads] || 8
    dim_head = opts[:dim_head] || 64
    dropout_rate = opts[:dropout] || 0.0

    inner_dim = n_heads * dim_head
    scale = :math.sqrt(dim_head)

    # Create query, key, value projections using Axon layers
    query = input |> dense(inner_dim, name: "query")
    key = input |> dense(inner_dim, name: "key")
    value = input |> dense(inner_dim, name: "value")

    # Create attention computation using Axon's layer
    attention =
      layer(
        fn {q, k, v}, _opts ->
          # Reshape for multi-head attention
          batch_size = Nx.axis_size(q, 0)
          seq_len = Nx.axis_size(q, 1)

          reshape_heads = fn x ->
            x
            |> Nx.reshape({batch_size, seq_len, n_heads, dim_head})
            |> Nx.transpose(axes: [0, 2, 1, 3])
          end

          q = reshape_heads.(q)
          k = reshape_heads.(k)
          v = reshape_heads.(v)

          # Transpose key for dot product
          k = Nx.transpose(k, axes: [0, 1, 3, 2])

          # Compute scaled dot-product attention
          # Shape: (batch, n_heads, seq_len, seq_len)
          attention_scores =
            Nx.dot(q, k)
            |> Nx.divide(scale)

          # Apply softmax
          attention_scores = Axon.activation(attention_scores, :softmax)

          # Apply attention to values
          attention_output =
            Nx.dot(attention_scores, v)
            |> Nx.transpose(axes: [0, 2, 1, 3])
            |> Nx.reshape({batch_size, seq_len, inner_dim})

          attention_output
        end,
        [{query, key, value}],
        name: "attention_computation"
      )

    attention
    |> dense(dim, name: "output_projection")
    |> dropout(rate: dropout_rate)
  end
end
