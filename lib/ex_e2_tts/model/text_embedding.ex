defmodule ExE2Tts.Model.TextEmbedding do
  @moduledoc """
  Implements text embedding and processing for the TTS model.
  """

  import Axon

  @doc """
  Creates text embedding layers with optional convolutional processing.

  ## Parameters
    * input - The input layer to build upon
    * opts - Options for configuring the embedding
      * vocab_size - Size of the vocabulary (required)
      * dim - Embedding dimension (default: 512)
      * conv_layers - Number of convolutional layers to apply (default: 0)
  """
  def create(input, opts \\ []) do
    vocab_size = opts[:vocab_size] || raise ArgumentError, "vocab_size is required"
    dim = opts[:dim] || 512
    conv_layers = opts[:conv_layers] || 0

    # Create embedding layer
    embedded = Axon.embedding(input, vocab_size, dim, name: "text_embedding")

    # Add convolutional layers if specified
    if conv_layers > 0 do
      build_conv_layers(embedded, dim, conv_layers)
    else
      embedded
    end
  end

  defp build_conv_layers(input, dim, num_layers) do
    Enum.reduce(1..num_layers, input, fn i, acc ->
      acc
      |> conv(dim,
        kernel_size: 3,
        padding: :same,
        name: "conv_#{i}"
      )
      |> batch_norm(name: "bn_#{i}")
      |> activation(:mish)
    end)
  end
end
