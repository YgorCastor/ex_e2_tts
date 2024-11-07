defmodule ExE2Tts.Model.Attention do
  @moduledoc """
  Implements scaled dot-product attention with support for rotary embeddings
  and classifier-free guidance.
  """

  import Axon
  import Nx.Defn

  @doc """
  Creates a multi-head attention layer using Axon.

  ## Parameters
    * input - The input layer to build upon
    * opts - Configuration options:
      * dim - Model dimension (must be divisible by heads)
      * heads - Number of attention heads 
      * dim_head - Dimension per head 
      * dropout - Dropout rate
      * context_dim - Optional dimension for cross-attention
  """
  def create(input, opts \\ []) do
    # Get configuration
    dim = opts[:dim] || 512
    heads = opts[:heads] || 8
    dim_head = opts[:dim_head] || div(dim, heads)
    dropout_rate = opts[:dropout] || 0.0
    context_dim = opts[:context_dim]

    inner_dim = heads * dim_head
    scale = :math.sqrt(dim_head)

    # Input projections using Axon layers
    query = create_projection(input, inner_dim, "query")

    # Handle context for cross-attention if needed
    {key, value} =
      if context_dim do
        context = opts[:context]
        key = create_projection(context, inner_dim, "key")
        value = create_projection(context, inner_dim, "value")
        {key, value}
      else
        key = create_projection(input, inner_dim, "key")
        value = create_projection(input, inner_dim, "value")
        {key, value}
      end

    # Combine inputs into attention layer
    attention =
      create_attention_layer(
        {query, key, value},
        heads: heads,
        dim_head: dim_head,
        scale: scale,
        dropout_rate: dropout_rate
      )

    # Output projection and normalization
    attention
    |> create_output_layers(dim, dropout_rate)
  end

  defp create_projection(input, dim, name) do
    input
    |> dense(dim,
      name: name,
      kernel_initializer: :glorot_uniform,
      use_bias: true
    )
    # Keep linear activation for QKV projections
    |> activation(:linear)
  end

  defp create_attention_layer({query, key, value}, opts) do
    layer(
      fn {q, k, v}, layer_opts ->
        {batch_size, seq_len, _} = Nx.shape(q)

        queries = reshape_qkv(q, batch_size, seq_len, opts[:heads], opts[:dim_head])
        keys = reshape_qkv(k, batch_size, seq_len, opts[:heads], opts[:dim_head])
        values = reshape_qkv(v, batch_size, seq_len, opts[:heads], opts[:dim_head])

        # Apply rotary embeddings if provided
        {queries, keys} = apply_rotary_emb(queries, keys, layer_opts[:rope])

        compute_scaled_attention(
          queries,
          keys,
          values,
          opts[:scale],
          layer_opts[:mask],
          opts[:dropout_rate]
        )
      end,
      [{query, key, value}],
      name: "attention_computation"
    )
  end

  defp create_output_layers(attention, dim, dropout_rate) do
    attention
    |> dense(dim,
      name: "output_projection",
      kernel_initializer: :glorot_uniform,
      use_bias: true
    )
    |> dropout(rate: dropout_rate)
  end

  # Helper function for QKV reshaping using Axon operations
  defp reshape_qkv(x, batch_size, seq_len, heads, dim_head) do
    x
    |> reshape({batch_size, seq_len, heads, dim_head})
    |> transpose(axes: [0, 2, 1, 3])
  end

  # Apply rotary embeddings through Axon layer
  defp apply_rotary_emb(query, key, rope) do
    if rope != nil do
      {freqs, xpos_scale} = rope

      # Create a dedicated layer for rotary embeddings
      rotary_layer =
        ExE2Tts.Model.RotaryEmbedding.create_layer(
          dim: query.shape[-1],
          scale: xpos_scale
        )

      query = rotary_layer.(query, freqs)
      key = rotary_layer.(key, freqs)
      {query, key}
    else
      {query, key}
    end
  end

  defp compute_scaled_attention(query, key, value, scale, mask, dropout_rate) do
    layer(
      fn inputs, _opts ->
        compute_attn(inputs.query, inputs.key, inputs.value, scale, mask, dropout_rate)
      end,
      [%{query: query, key: key, value: value}],
      name: "scaled_dot_product"
    )
  end

  defn compute_attn(q, k, v, s, m, d) do
    scores =
      Nx.dot(q, Nx.transpose(k))
      |> Nx.divide(s)

    # Apply mask if provided
    scores =
      if m != nil do
        Nx.select(
          m,
          scores,
          Nx.broadcast(-1.0e9, Nx.shape(scores))
        )
      else
        scores
      end

    probs = Axon.Activations.softmax(scores)

    # Dropout
    probs =
      if d > 0 do
        dropout_mask = Nx.Random.uniform(Nx.shape(probs)) < d
        Nx.select(dropout_mask, Nx.broadcast(0.0, Nx.shape(probs)), probs)
      else
        probs
      end

    # Final attention computation
    Nx.dot(probs, v)
  end

  @doc """
  Creates a cross-attention variant of the attention layer.
  """
  def create_cross_attention(input, context, opts \\ []) do
    opts = Keyword.merge(opts, context_dim: context.shape[-1], context: context)
    create(input, opts)
  end

  @doc """
  Creates a self-attention variant with optional causal masking.
  """
  def create_self_attention(input, opts \\ []) do
    causal = Keyword.get(opts, :causal, false)

    if causal do
      opts = add_causal_mask(opts, input)
      create(input, opts)
    else
      create(input, opts)
    end
  end

  # Helper to add causal masking for self-attention
  defp add_causal_mask(opts, input) do
    mask_layer =
      layer(
        fn x, _opts ->
          {_, seq_len, _} = Nx.shape(x)
          mask = Nx.triu(Nx.broadcast(1, {seq_len, seq_len}), k: 1)
          Nx.equal(mask, 0)
        end,
        [input]
      )

    Keyword.put(opts, :mask, mask_layer)
  end
end
