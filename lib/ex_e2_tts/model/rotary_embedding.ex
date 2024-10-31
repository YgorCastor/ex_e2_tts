defmodule ExE2Tts.Model.RotaryEmbedding do
  @moduledoc """
  Implements Rotary Position Embeddings (RoPE) with caching mechanism.

  RoPE performs rotation in the complex plane based on position, enabling better
  relative positional awareness in transformer models. This implementation includes
  both the core Nx functionality and Axon-compatible layers.
  """
  @default_theta 10_000.0
  @default_theta_rescale_factor 1.0

  import Nx.Defn
  import Axon

  defstruct [
    :dim,
    :freqs_cis,
    :max_position,
    :theta,
    :theta_rescale_factor
  ]

  @doc """
  Creates a new RotaryEmbedding instance.
  Initial cache for sequences up to max_position length.

  ## Options
    * dim - Embedding dimension (must be even)
    * max_position - Maximum sequence length to cache (default: 4096)
    * theta - Base for frequency bands (default: 10_000.0)
    * theta_rescale_factor - Rescaling factor for longer sequences (default: 1.0)
  """
  def new(dim, opts \\ []) do
    max_position = opts[:max_position] || 4096
    theta = opts[:theta] || @default_theta
    theta_rescale_factor = opts[:theta_rescale_factor] || @default_theta_rescale_factor

    # Validate dim is even
    unless rem(dim, 2) == 0 do
      raise ArgumentError, "dimension must be divisible by 2, got: #{dim}"
    end

    # Initialize with cached frequency computations
    freqs_cis = precompute_freqs_cis(dim, max_position, theta, theta_rescale_factor)

    %__MODULE__{
      dim: dim,
      max_position: max_position,
      freqs_cis: freqs_cis,
      theta: theta,
      theta_rescale_factor: theta_rescale_factor
    }
  end

  @doc """
  Creates an Axon layer that applies rotary embeddings.

  ## Options
    * dim - Head dimension (must be even, equals head_dim)
    * max_position - Maximum sequence length to cache (default: 4096)
    * rescale_factor - Position scaling factor (default: 1.0)

  ## Shape expectations
  Input: [batch, sequence, heads, head_dim] where head_dim must equal dim
  """
  def create_layer(input, opts \\ []) do
    dim = opts[:dim] || raise ArgumentError, "dim is required"
    max_position = opts[:max_position] || 4096
    rescale_factor = opts[:rescale_factor] || 1.0

    layer(
      fn x, _opts ->
        # Get sequence length from input shape
        {_batch, seq_len, _n_heads, head_dim} = Nx.shape(x)

        # Validate dimensions
        unless head_dim == dim do
          raise ArgumentError,
                "Input head dimension #{head_dim} does not match provided dim #{dim}"
        end

        # Create rotary embedding instance
        rotary = new(dim, max_position)

        # Get frequencies for this sequence length
        {freqs, _} = forward_from_seq_len(rotary, seq_len)

        # Apply embeddings
        apply_rotary_pos_emb(x, freqs, rescale_factor)
      end,
      [input],
      name: "rotary_embedding"
    )
  end

  # Precomputes frequency bands and their complex components.
  # Uses rescaling technique for better handling of longer sequences.
  defnp precompute_freqs_cis(dim, max_position, theta, theta_rescale_factor) do
    # Rescale theta for longer sequences using NTK insights
    theta = scale_theta(theta, theta_rescale_factor, dim)

    # Calculate frequency bands
    half_dim = div(dim, 2)
    freqs = compute_frequency_bands(theta, dim, half_dim)

    # Generate position indices and compute outer product
    t = Nx.iota({max_position}, type: :f32)
    freqs = Nx.outer(t, freqs)

    # Generate complex components
    {freqs_cos, freqs_sin} = generate_complex_components(freqs)

    # Stack for caching
    Nx.stack([freqs_cos, freqs_sin], axis: 0)
  end

  defnp scale_theta(theta, scale_factor, dim) do
    Nx.multiply(
      theta,
      Nx.pow(
        scale_factor,
        Nx.divide(dim, Nx.subtract(dim, 2))
      )
    )
  end

  defnp compute_frequency_bands(theta, dim, half_dim) do
    Nx.divide(
      1.0,
      Nx.pow(
        theta,
        Nx.divide(
          Nx.take(Nx.iota({dim}), Nx.iota({half_dim}) * 2),
          dim
        )
      )
    )
  end

  defnp generate_complex_components(freqs) do
    freqs_cos = Nx.cos(freqs)
    freqs_sin = Nx.sin(freqs)
    {freqs_cos, freqs_sin}
  end

  # Gets cached frequencies for a given sequence length.
  # Automatically handles cache expansion if needed.
  def get_freqs(rotary, seq_len) do
    if seq_len > rotary.max_position do
      new_max_pos = max(seq_len * 2, rotary.max_position * 2)

      freqs_cis =
        precompute_freqs_cis(
          rotary.dim,
          new_max_pos,
          rotary.theta,
          rotary.theta_rescale_factor
        )

      rotary = %{rotary | freqs_cis: freqs_cis, max_position: new_max_pos}
      {Nx.slice(rotary.freqs_cis, [0], [seq_len]), rotary}
    else
      {Nx.slice(rotary.freqs_cis, [0], [seq_len]), rotary}
    end
  end

  @doc """
  Forward pass computing rotary embeddings for a given sequence length.
  Returns cached frequencies and optional xpos scale factor.

  ## Parameters
    * rotary - Rotary embedding instance
    * seq_len - Sequence length
    * xpos_scale - Optional scaling factor for positions (default: nil)
  """
  def forward_from_seq_len(rotary, seq_len, xpos_scale \\ nil) do
    {freqs, _rotary} = get_freqs(rotary, seq_len)
    {freqs, xpos_scale}
  end

  @doc """
  Applies rotary embeddings to input tensor using complex rotations.

  ## Parameters
    * tensor - Input tensor of shape [batch, seq_len, n_heads, head_dim]
    * freqs - Precomputed frequencies
    * scale - Optional scaling factor (default: 1.0)
  """
  defn apply_rotary_pos_emb(tensor, freqs, scale \\ 1.0) do
    {_batch, seq_len, n_heads, head_dim} = Nx.shape(tensor)
    half_head_dim = div(head_dim, 2)

    # Split tensor for complex rotation
    {tensor_1, tensor_2} = split_complex(tensor, seq_len, n_heads, half_head_dim)

    # Get frequency components
    {freqs_cos, freqs_sin} = prepare_frequencies(freqs, seq_len, half_head_dim)

    # Apply complex rotation
    {rotated_1, rotated_2} =
      apply_complex_rotation(
        tensor_1,
        tensor_2,
        freqs_cos,
        freqs_sin
      )

    # Combine and rescale
    combine_and_scale(rotated_1, rotated_2, tensor, scale)
  end

  defnp split_complex(tensor, seq_len, n_heads, half_head_dim) do
    [tensor_1, tensor_2] =
      tensor
      |> Nx.reshape({-1, seq_len, n_heads, 2, half_head_dim})
      |> Nx.split(2, axis: 3)

    {tensor_1, tensor_2}
  end

  defnp prepare_frequencies(freqs, seq_len, half_head_dim) do
    freqs
    |> Nx.slice([0, 0], [seq_len, half_head_dim])
    |> Nx.reshape({seq_len, 1, half_head_dim})
    |> Nx.split(2)
  end

  defnp apply_complex_rotation(tensor_1, tensor_2, freqs_cos, freqs_sin) do
    rotated_1 = Nx.multiply(tensor_1, freqs_cos) - Nx.multiply(tensor_2, freqs_sin)
    rotated_2 = Nx.multiply(tensor_1, freqs_sin) + Nx.multiply(tensor_2, freqs_cos)
    {rotated_1, rotated_2}
  end

  defnp combine_and_scale(rotated_1, rotated_2, original_tensor, scale) do
    {batch, seq_len, n_heads, head_dim} = Nx.shape(original_tensor)

    Nx.concatenate([rotated_1, rotated_2], axis: 3)
    |> Nx.reshape({batch, seq_len, n_heads, head_dim})
    |> Nx.multiply(scale)
  end
end
