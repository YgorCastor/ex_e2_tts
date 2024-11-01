defmodule ExE2Tts.Dataset do
  @moduledoc """
  Dataset implementation for E2/F5 TTS training.
  Assumes audio files are already in correct format (24kHz, mono).
  """

  alias ExE2Tts.Dataset.Audio
  alias ExE2Tts.MelSpec

  defstruct [
    # Raw dataset entries
    :data,
    :target_sample_rate,
    :n_mel_channels,
    :hop_length,
    :durations,
    :preprocessed_mel,
    :mel_spec_module,
    :cache
  ]

  @doc """
  Creates a new dataset instance with the specified options.

  ## Options
    * :target_sample_rate - Target sample rate (default: 24000)
    * :n_mel_channels - Number of mel channels (default: 100)
    * :hop_length - Hop length for STFT (default: 256)
    * :preprocessed_mel - Whether mel specs are pre-computed (default: false)
    * :mel_spec_module - Custom mel spectrogram processor (optional)
    * :durations - List of pre-computed durations (optional)

  ## Example
      iex> data = [
      ...>   %{"audio_path" => "path/to/audio1.wav", "text" => "Sample text 1"},
      ...>   %{"audio_path" => "path/to/audio2.wav", "text" => "Sample text 2"}
      ...> ]
      iex> Dataset.new(data, preprocessed_mel: false)
  """
  def new(data, opts \\ []) do
    opts = Keyword.merge(default_audio_options(), opts)

    mel_spec_module =
      case opts[:mel_spec_module] do
        nil ->
          MelSpec.new(
            target_sample_rate: opts[:target_sample_rate],
            n_mel_channels: opts[:n_mel_channels],
            hop_length: opts[:hop_length]
          )

        module ->
          module
      end

    cache =
      if opts[:cache] do
        ExE2Tts.Dataset.Cache.new(
          cache_type: opts[:cache_type] || :memory,
          cache_dir: opts[:cache_dir],
          max_memory_items: opts[:max_memory_items]
        )
      end

    %__MODULE__{
      data: data,
      target_sample_rate: opts[:target_sample_rate],
      n_mel_channels: opts[:n_mel_channels],
      hop_length: opts[:hop_length],
      durations: opts[:durations],
      preprocessed_mel: opts[:preprocessed_mel],
      mel_spec_module: mel_spec_module,
      cache: cache
    }
  end

  @doc """
  Creates a dataloader with dynamic batch sampling.

  ## Options
    * :frames_threshold - Maximum frames per batch
    * :max_samples - Maximum samples per batch
    * :random_seed - Seed for shuffling
    * :drop_last - Whether to drop last incomplete batch
  """
  def dataloader(%__MODULE__{} = dataset, opts \\ []) do
    sampler = ExE2Tts.Dataset.DynamicBatchSampler.new(dataset, opts)

    Stream.map(ExE2Tts.Dataset.DynamicBatchSampler.stream(sampler), fn batch_indices ->
      # Fetch and collate batch items
      items = Enum.map(batch_indices, &fetch(dataset, &1))
      collate_batch(items)
    end)
  end

  @doc """
  Gets the frame length for dynamic batch sampling.
  Uses pre-computed durations if available, otherwise reads from data entry.
  """
  def get_frame_len(%__MODULE__{} = dataset, index) do
    case dataset.durations do
      nil ->
        item = Enum.at(dataset.data, index)
        # Assuming duration is stored in data
        item["duration"] * dataset.target_sample_rate / dataset.hop_length

      durations ->
        # Use pre-computed duration
        Enum.at(durations, index) * dataset.target_sample_rate / dataset.hop_length
    end
  end

  @doc """
  Returns the total number of samples in the dataset.
  """
  def size(%__MODULE__{data: data}), do: length(data)

  @doc """
  Fetches a single item from the dataset.
  Returns a map with :mel_spec and :text keys.

  ## Example
      iex> dataset |> Dataset.fetch(0)
      %{
        mel_spec: #Nx.Tensor<...>,
        text: "Sample text"
      }
  """
  def fetch(%__MODULE__{cache: nil} = dataset, index) when is_integer(index) do
    item = Enum.at(dataset.data, index)

    mel_spec =
      if dataset.preprocessed_mel do
        # Load pre-computed mel spectrogram
        item["mel_spec"]
      else
        audio_tensor = Audio.load_wav(item["audio_path"])
        process_audio_to_mel(dataset, audio_tensor)
      end

    %{
      mel_spec: mel_spec,
      text: item["text"]
    }
  end

  def fetch(%__MODULE__{cache: cache} = dataset, index) do
    item = Enum.at(dataset.data, index)

    if dataset.preprocessed_mel do
      # Direct return for preprocessed data
      %{
        mel_spec: item["mel_spec"],
        text: item["text"]
      }
    else
      # Use cache for mel spectrograms
      cache_key = {"mel_spec", item["audio_path"]}

      {:ok, mel_spec} =
        ExE2Tts.Dataset.Cache.get_or_store(cache, cache_key, fn ->
          audio = Audio.load_wav(item["audio_path"])
          process_audio_to_mel(dataset, audio)
        end)

      %{
        mel_spec: mel_spec,
        text: item["text"]
      }
    end
  end

  def validate_audio_files!(data) do
    Enum.each(data, fn item ->
      path = item["audio_path"]

      unless File.exists?(path) do
        raise "Audio file not found: #{path}"
      end

      # Try loading first few files to validate format
      # Sample 10% of files
      if :rand.uniform() < 0.1 do
        Audio.load_wav(path)
      end
    end)
  end

  defp collate_batch(items) do
    # Separate mel specs and texts
    {mel_specs, texts} = separate_items(items)

    # Process mel spectrograms
    {mel_batch, mel_lengths} = process_mel_specs(mel_specs)

    # Process text lengths
    text_lengths = process_text_lengths(texts)

    %{
      mel: mel_batch,
      mel_lengths: mel_lengths,
      text: texts,
      text_lengths: text_lengths
    }
  end

  defp separate_items(items) do
    items
    |> Enum.map(fn item -> {item.mel_spec, item.text} end)
    |> Enum.unzip()
  end

  defp process_mel_specs(mel_specs) do
    # Get max length for padding
    max_len = get_max_length(mel_specs)

    # Pad and stack spectrograms
    mel_batch =
      mel_specs
      |> pad_spectrograms(max_len)
      |> Nx.stack()

    # Get original lengths
    mel_lengths = Nx.tensor(Enum.map(mel_specs, &elem(Nx.shape(&1), 1)))

    {mel_batch, mel_lengths}
  end

  defp get_max_length(mel_specs) do
    mel_specs
    |> Enum.map(&elem(Nx.shape(&1), 1))
    |> Enum.max()
  end

  defp pad_spectrograms(mel_specs, max_len) do
    Enum.map(mel_specs, fn mel ->
      pad_spectrogram(mel, max_len)
    end)
  end

  defp pad_spectrogram(mel, max_len) do
    padding = max_len - elem(Nx.shape(mel), 1)

    if padding > 0 do
      Nx.pad(mel, [{0, 0}, {0, padding}], value: 0)
    else
      mel
    end
  end

  defp process_text_lengths(texts) do
    texts
    |> Enum.map(&String.length/1)
    |> Nx.tensor()
  end

  defp default_audio_options do
    [
      target_sample_rate: 24_000,
      n_mel_channels: 100,
      hop_length: 256,
      preprocessed_mel: false,
      mel_spec_module: nil,
      durations: nil
    ]
  end

  defp process_audio_to_mel(dataset, audio) do
    # Convert to mel spectrogram
    audio
    # Add batch dimension
    |> Nx.new_axis(0)
    |> MelSpec.audio_to_mel(dataset.mel_spec_module)
    # Remove batch dimension
    |> Nx.squeeze(0)
  end
end
