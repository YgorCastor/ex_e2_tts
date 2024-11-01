defmodule ExE2Tts.Dataset.DynamicBatchSampler do
  @moduledoc """
  Dynamic batch sampler that creates batches based on sequence lengths.
  Groups similarly-sized sequences together to minimize padding while 
  keeping total frames per batch under a threshold.
  """

  defstruct [
    :dataset,
    :frames_threshold,
    :max_samples,
    :batches,
    :random_seed
  ]

  @type t :: %__MODULE__{
          dataset: ExE2Tts.Dataset.t(),
          frames_threshold: pos_integer(),
          max_samples: non_neg_integer(),
          batches: list(list(non_neg_integer())),
          random_seed: integer() | nil
        }

  @doc """
  Creates a new dynamic batch sampler.

  ## Options
    * frames_threshold - Maximum total frames per batch
    * max_samples - Maximum sequences per batch (default: 0, no limit)
    * random_seed - Seed for batch shuffling (optional)
    * drop_last - Whether to drop last incomplete batch (default: false)
  """
  def new(dataset, opts \\ []) do
    frames_threshold =
      opts[:frames_threshold] || raise ArgumentError, "frames_threshold is required"

    max_samples = opts[:max_samples] || 0
    drop_last = opts[:drop_last] || false
    random_seed = opts[:random_seed]

    # Create batches
    batches = create_batches(dataset, frames_threshold, max_samples, drop_last)

    # Shuffle if seed provided
    batches =
      if random_seed do
        :rand.seed(:exs1024, {random_seed, 0, 0})
        Enum.shuffle(batches)
      else
        batches
      end

    %__MODULE__{
      dataset: dataset,
      frames_threshold: frames_threshold,
      max_samples: max_samples,
      batches: batches,
      random_seed: random_seed
    }
  end

  @doc """
  Returns total number of batches.
  """
  def num_batches(%__MODULE__{batches: batches}), do: length(batches)

  @doc """
  Returns iterator for batch indices.
  """
  def stream(%__MODULE__{batches: batches}), do: Stream.map(batches, & &1)

  defp create_batches(dataset, frames_threshold, max_samples, drop_last) do
    dataset
    |> get_indexed_frame_lengths()
    |> sort_by_frame_length()
    |> create_batches_from_sorted(frames_threshold, max_samples, drop_last)
  end

  defp get_indexed_frame_lengths(dataset) do
    0..(ExE2Tts.Dataset.size(dataset) - 1)
    |> Enum.map(fn idx ->
      frame_len = ExE2Tts.Dataset.get_frame_len(dataset, idx)
      {idx, frame_len}
    end)
  end

  defp sort_by_frame_length(indices_with_lengths) do
    Enum.sort_by(indices_with_lengths, fn {_idx, len} -> len end)
  end

  defp create_batches_from_sorted(sorted_indices, frames_threshold, max_samples, drop_last) do
    # {batches, current_batch, current_frames}
    initial_state = {[], [], 0}

    {batches, current_batch, _} =
      Enum.reduce(sorted_indices, initial_state, fn sample, acc ->
        process_sample(sample, acc, frames_threshold, max_samples)
      end)

    finalize_batches(batches, current_batch, drop_last)
  end

  defp process_sample(
         {idx, frame_len},
         {batches, current_batch, current_frames},
         frames_threshold,
         max_samples
       ) do
    cond do
      can_add_to_current_batch?(
        frame_len,
        current_frames,
        current_batch,
        frames_threshold,
        max_samples
      ) ->
        add_to_current_batch(batches, current_batch, current_frames, idx, frame_len)

      can_start_new_batch?(frame_len, frames_threshold) ->
        start_new_batch(batches, current_batch, idx, frame_len)

      true ->
        skip_oversized_sample(
          idx,
          frame_len,
          frames_threshold,
          batches,
          current_batch,
          current_frames
        )
    end
  end

  defp can_add_to_current_batch?(
         frame_len,
         current_frames,
         current_batch,
         frames_threshold,
         max_samples
       ) do
    current_frames + frame_len <= frames_threshold and
      (max_samples == 0 or length(current_batch) < max_samples)
  end

  defp add_to_current_batch(batches, current_batch, current_frames, idx, frame_len) do
    {batches, [idx | current_batch], current_frames + frame_len}
  end

  defp can_start_new_batch?(frame_len, frames_threshold) do
    frame_len <= frames_threshold
  end

  defp start_new_batch(batches, current_batch, idx, frame_len) do
    if current_batch == [] do
      {batches, [idx], frame_len}
    else
      {[Enum.reverse(current_batch) | batches], [idx], frame_len}
    end
  end

  defp skip_oversized_sample(
         idx,
         frame_len,
         frames_threshold,
         batches,
         current_batch,
         current_frames
       ) do
    IO.warn("Sample #{idx} exceeds frames_threshold (#{frame_len} > #{frames_threshold})")
    {batches, current_batch, current_frames}
  end

  defp finalize_batches(batches, current_batch, drop_last) do
    final_batches =
      if current_batch != [] and not drop_last do
        [Enum.reverse(current_batch) | batches]
      else
        batches
      end

    Enum.reverse(final_batches)
  end
end
