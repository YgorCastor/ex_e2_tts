defmodule ExE2Tts.TextUtils do
  @moduledoc """
  Utility functions supporting the CFM implementation.
  """

  import Nx.Defn

  def default_lens(lens, batch_size, seq_len, device) do
    if lens do
      lens
    else
      Nx.broadcast(seq_len, {batch_size}, type: :s64, device: device)
    end
  end

  defn lens_to_mask(lens) do
    seq_len = Nx.reduce_max(lens)
    seq = Nx.iota({seq_len})
    Nx.less(seq, lens)
  end

  def process_text_input(text, vocab_map, lens) when is_list(text) do
    # Convert text list to tensor using vocab map
    if vocab_map do
      text_tensor = text_to_indices(text, vocab_map)
      text_lens = Nx.sum(Nx.not_equal(text_tensor, -1), axes: [-1])
      lens = Nx.max(text_lens, lens)
      {text_tensor, lens}
    else
      text_tensor = text_to_bytes(text)
      text_lens = Nx.sum(Nx.not_equal(text_tensor, -1), axes: [-1])
      lens = Nx.max(text_lens, lens)
      {text_tensor, lens}
    end
  end

  def process_text_input(text, _vocab_map, lens) do
    {text, lens}
  end

  @doc """
  Converts a list of text strings to indices using a vocabulary map.
  Pads sequences with -1 to match the longest sequence in the batch.
  """
  def text_to_indices(text_list, vocab_map) do
    # Convert each text string to indices
    indices_list = Enum.map(text_list, fn text ->
      String.graphemes(text)
      |> Enum.map(fn char -> 
        Map.get(vocab_map, char, 0) # Use 0 for unknown characters
      end)
    end)

    # Find max length for padding
    max_len = indices_list |> Enum.map(&length/1) |> Enum.max()

    # Pad sequences with -1 and convert to tensor
    indices_list
    |> Enum.map(fn indices ->
      indices ++ List.duplicate(-1, max_len - length(indices))
    end)
    |> Nx.tensor()
  end

  @doc """
  Converts a list of text strings to UTF-8 byte tensors.
  Pads sequences with -1 to match the longest sequence in the batch.
  """
  def text_to_bytes(text_list) do
    # Convert each text string to UTF-8 bytes
    byte_lists = Enum.map(text_list, fn text ->
      :binary.bin_to_list(text)
    end)

    # Find max length and pad sequences
    max_len = byte_lists |> Enum.map(&length/1) |> Enum.max()

    byte_lists
    |> Enum.map(fn bytes ->
      bytes ++ List.duplicate(-1, max_len - length(bytes))
    end)
    |> Nx.tensor()
  end

  @doc """
  Gets the device of a tensor through backend transfer.
  """
  def device(tensor) do
    Nx.backend_transfer(tensor).device
  end
end
