defmodule ExE2Tts.Dataset.Audio do
  @moduledoc """
  Handles audio file loading and conversion to tensors.
  Currently supports only WAV files in PCM format.
  """
  @pcm_format 1

  @doc """
  Loads a WAV file and returns an Nx tensor.
  Assumes 24kHz, mono, 16-bit PCM format.
  """
  def load_wav(path) do
    case File.read(path) do
      {
        :ok,
        # ChunkID
        "RIFF",
        <<_chunk_size::32-little>>,
        # Format
        "WAVE",
        # Subchunk1ID 
        "fmt ",
        # Subchunk1Size (16 for PCM)
        <<16::32-little>>,
        # AudioFormat (1 for PCM)
        <<format::16-little>>,
        # NumChannels
        <<channels::16-little>>,
        # SampleRate
        <<sample_rate::32-little>>,
        # ByteRate
        <<_byte_rate::32-little>>,
        # BlockAlign
        <<_block_align::16-little>>,
        # BitsPerSample
        <<bits_per_sample::16-little>>,
        # Subchunk2ID
        "data",
        # Subchunk2Size
        <<_data_size::32-little>>,
        # The actual audio data
        <<audio_data::binary>>
      } ->
        validate_format!(format, channels, sample_rate, bits_per_sample)
        samples = parse_audio_data(audio_data, bits_per_sample)
        normalize_audio(samples)

      {:ok, _} ->
        raise "Invalid WAV file format"

      {:error, reason} ->
        raise "Failed to read audio file: #{reason}"
    end
  end

  # Validates audio format matches our expectations
  defp validate_format!(format, channels, sample_rate, bits_per_sample) do
    unless format == @pcm_format and
             channels == 1 and
             sample_rate == 24_000 and
             bits_per_sample == 16 do
      raise "Unsupported WAV format. Expected: 24kHz mono 16-bit PCM, " <>
              "Got: #{sample_rate}Hz #{channels}-channel #{bits_per_sample}-bit"
    end
  end

  # Parses audio data based on bits per sample
  defp parse_audio_data(data, bits_per_sample) do
    case bits_per_sample do
      16 ->
        # Convert 16-bit PCM samples to list of integers
        for <<sample::16-signed-little <- data>>, do: sample

      _ ->
        raise "Unsupported bits per sample: #{bits_per_sample}"
    end
  end

  # Normalizes audio samples to float32 tensor in range [-1, 1]
  defp normalize_audio(samples) do
    samples
    |> Nx.tensor(type: {:s, 16})
    # 2^15 for 16-bit audio
    |> Nx.divide(32_768.0)
    |> Nx.as_type({:f, 32})
  end
end
