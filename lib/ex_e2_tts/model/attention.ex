defmodule ExE2Tts.Model.Attention do
  @moduledoc """
  Implements scaled dot-product attention with support for rotary embeddings
  and classifier-free guidance.
  """

  import Nx.Defn
  import Axon

  @doc """
  Creates a multi-head attention layer with specified dimensions.

  ## Parameters
    * input - The input layer to build upon
    * opts - Options for configuring the attention mechanism:
      * dim - Model dimension (must be divisible by heads)
      * heads - Number of attention heads
      * dim_head - Dimension per head (default: dim/heads)
      * dropout - Dropout rate
      * context_dim - Optional dimension for cross-attention
  """
  def create(input, opts \\ []) do
    dim = opts[:dim] || 512
    heads = opts[:heads] || 8
    dim_head = opts[:dim_head] || div(dim, heads)
    dropout_rate = opts[:dropout] || 0.0
    context_dim = opts[:context_dim]

    inner_dim = heads * dim_head
    scale = :math.sqrt(dim_head)

    # Create query, key, value projections
    query = input |> dense(inner_dim, name: "query")
    key = input |> dense(inner_dim, name: "key")
    value = input |> dense(inner_dim, name: "value")

    # Optional cross-attention
    {key, value} =
      if context_dim do
        context = opts[:context]
        key_cross = context |> dense(inner_dim, name: "key_cross")
        value_cross = context |> dense(inner_dim, name: "value_cross")
        {key_cross, value_cross}
      else
        {key, value}
      end

    # Core attention computation
    attention =
      layer(
        fn {q, k, v}, opts ->
          # Get shape information
          {batch_size, seq_len, _} = Nx.shape(q)

          # Get mask and rope from opts if provided
          mask = opts[:mask]
          rope = opts[:rope]

          # Process mask if provided
          mask =
            if mask != nil do
              mask
              |> Nx.new_axis(1)
              |> Nx.new_axis(1)
              |> Nx.broadcast({batch_size, 1, 1, seq_len})
            end

          # Reshape for multi-head attention
          query = reshape_for_attention(q, batch_size, seq_len, heads, dim_head)
          key = reshape_for_attention(k, batch_size, seq_len, heads, dim_head)
          value = reshape_for_attention(v, batch_size, seq_len, heads, dim_head)

          # Apply rotary embeddings if provided
          {query, key} = apply_rotary_embeddings(query, key, rope)

          # Scaled dot-product attention with masking
          attention_scores = compute_attention_scores(query, key, scale, mask)

          # Apply softmax and dropout
          attention_scores =
            attention_scores
            |> Axon.Activations.softmax()
            |> dropout(rate: dropout_rate)

          # Apply attention to values with masking
          attended = apply_attention(attention_scores, value, mask)

          # Reshape back
          attended
          |> Nx.transpose(axes: [0, 2, 1, 3])
          |> Nx.reshape({batch_size, seq_len, inner_dim})
        end,
        [{query, key, value}],
        name: "attention_computation"
      )

    # Output projection
    attention
    |> dense(dim, name: "output_projection")
    |> dropout(rate: dropout_rate)
  end

  defp apply_rotary_embeddings(query, key, rope) do
    if rope != nil do
      {freqs, xpos_scale} = rope
      q_xpos_scale = if xpos_scale != nil, do: xpos_scale, else: 1.0
      k_xpos_scale = if xpos_scale != nil, do: 1.0 / xpos_scale, else: 1.0

      query = ExE2Tts.Model.RotaryEmbedding.apply_rotary_pos_emb(query, freqs, q_xpos_scale)
      key = ExE2Tts.Model.RotaryEmbedding.apply_rotary_pos_emb(key, freqs, k_xpos_scale)
      {query, key}
    else
      {query, key}
    end
  end

  defnp reshape_for_attention(x, batch_size, seq_len, heads, dim_head) do
    x
    |> Nx.reshape({batch_size, seq_len, heads, dim_head})
    |> Nx.transpose(axes: [0, 2, 1, 3])
  end

  defnp compute_attention_scores(query, key, scale, mask \\ nil) do
    # query shape: [batch, heads, seq_q, dim_head]
    # key shape: [batch, heads, seq_k, dim_head]
    # mask shape: [batch, 1, 1, seq_k] or nil
    # output shape: [batch, heads, seq_q, seq_k]

    scores =
      Nx.dot(
        query,
        [3],
        [0, 1],
        key,
        [3],
        [0, 1]
      )
      |> Nx.divide(scale)

    if mask != nil do
      Nx.select(
        mask,
        scores,
        Nx.broadcast(-1.0e9, Nx.shape(scores))
      )
    else
      scores
    end
  end

  defnp apply_attention(attention_scores, value, mask \\ nil) do
    # attention_scores shape: [batch, heads, seq_q, seq_k]
    # value shape: [batch, heads, seq_k, dim_head]
    # mask shape: [batch, 1, 1, seq_k] or nil
    # output shape: [batch, heads, seq_q, dim_head]

    attended =
      Nx.dot(
        attention_scores,
        [3],
        [0, 1],
        value,
        [2],
        [0, 1]
      )

    if mask != nil do
      mask_expanded =
        Nx.broadcast(
          mask,
          Nx.shape(attended)
        )

      Nx.select(mask_expanded, attended, 0.0)
    else
      attended
    end
  end
end
