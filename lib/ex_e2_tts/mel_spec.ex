defmodule ExE2Tts.MelSpec do
  @moduledoc """
  Handles mel-spectrogram generation and processing.
  Provides functionality to convert raw audio waveforms to mel spectrograms
  using the Short-time Fourier transform (STFT) and mel filterbank.
  """

  import Nx.Defn

  defstruct [
    :n_mel_channels,
    :hop_length,
    :sample_rate,
    :filter_length,
    :win_length,
    :normalize,
    :power,
    :norm,
    :center,
    :mel_filterbank
  ]

  @doc """
  Creates a new MelSpec instance with configured parameters.

  ## Options
    * :n_mel_channels - Number of mel bands (default: 100)
    * :hop_length - Number of samples between STFT columns (default: 256)
    * :sample_rate - Audio sample rate (default: 24000)
    * :filter_length - FFT size (default: 1024)
    * :win_length - Window size (default: 1024)
    * :normalize - Whether to normalize mel spectrograms (default: false)
    * :power - Power of the magnitude (default: 1.0)
    * :norm - Normalization method (:slaney or nil) (default: nil)
    * :center - Whether to pad signal on both sides (default: true)
  """
  def new(opts \\ []) do
    opts =
      Keyword.merge(
        [
          n_mel_channels: 100,
          hop_length: 256,
          sample_rate: 24_000,
          filter_length: 1024,
          win_length: 1024,
          normalize: false,
          power: 1.0,
          norm: nil,
          center: true
        ],
        opts
      )

    # Create mel filterbank matrix
    mel_fb =
      create_mel_filterbank(
        opts[:filter_length],
        opts[:sample_rate],
        opts[:n_mel_channels],
        opts[:norm]
      )

    struct!(__MODULE__, Keyword.merge(opts, mel_filterbank: mel_fb))
  end

  @doc """
  Converts audio waveform to mel spectrogram.

  ## Parameters
    * audio - Input audio tensor of shape [batch, time] or [batch, 1, time]
  """
  defn audio_to_mel(mel_spec, audio) do
    # Handle input shape
    audio = if Nx.rank(audio) == 3, do: Nx.squeeze(audio, axes: [1]), else: audio

    # Apply STFT
    stft =
      compute_stft(
        audio,
        mel_spec.filter_length,
        mel_spec.hop_length,
        mel_spec.win_length,
        mel_spec.center
      )

    # Convert to magnitude
    magnitude = Nx.abs(stft)

    # Apply power
    magnitude =
      if mel_spec.power != 1.0 do
        Nx.pow(magnitude, mel_spec.power)
      else
        magnitude
      end

    # Apply mel filterbank
    mel = Nx.dot(magnitude, mel_spec.mel_filterbank)

    # Convert to log scale with clipping
    mel = Nx.log(Nx.max(mel, 1.0e-5))

    # Normalize if requested
    if mel_spec.normalize do
      mean = Nx.mean(mel, axes: [-1], keepdims: true)
      std = Nx.standard_deviation(mel, axes: [-1], keepdims: true)
      (mel - mean) / (std + 1.0e-5)
    else
      mel
    end
  end

  defnp compute_stft(audio, n_fft, hop_length, win_length, center) do
    # Window function
    window = create_hann_window(win_length)

    # Pad signal if center is true
    padded_audio =
      if center do
        pad_size = div(n_fft, 2)
        Nx.pad(audio, [{0, 0}, {pad_size, pad_size}], mode: :reflect)
      else
        audio
      end

    # Compute frame indices
    n_frames = div(Nx.axis_size(padded_audio, 1) - n_fft, hop_length) + 1

    # Create frames
    frames = create_frames(padded_audio, n_fft, hop_length, n_frames)

    # Apply window
    frames = frames * window

    # Compute FFT
    Nx.FFT.fft(frames, length: n_fft)
  end

  defnp create_frames(audio, frame_length, hop_length, n_frames) do
    batch_size = Nx.axis_size(audio, 0)

    # Create frame indices with proper new axis syntax
    frame_indices = Nx.iota({n_frames}) * hop_length
    sample_indices = Nx.iota({frame_length})

    # Add new axes for broadcasting
    # Add axis at position 1
    frame_indices_expanded = Nx.new_axis(frame_indices, 1)
    # Add axis at position 0
    sample_indices_expanded = Nx.new_axis(sample_indices, 0)

    indices = frame_indices_expanded + sample_indices_expanded

    # Gather frames
    Nx.take(audio, indices, axis: 1)
    |> Nx.reshape({batch_size, n_frames, frame_length})
  end

  defnp create_hann_window(length) do
    n = Nx.iota({length})
    0.5 - 0.5 * Nx.cos(2.0 * :math.pi() * n / (length - 1))
  end

  defnp create_mel_filterbank(n_fft, sample_rate, n_mels, norm) do
    # Convert Hz to mel scale
    min_mel = hz_to_mel(0.0)
    max_mel = hz_to_mel(sample_rate / 2.0)

    # Create mel points
    mels = Nx.linspace(min_mel, max_mel, n_mels + 2)
    freqs = mel_to_hz(mels)

    # Convert frequencies to FFT bins
    bins = Nx.round(freqs * (n_fft + 1) / sample_rate)

    # Create filterbank matrix
    fbank = create_triangular_filters(bins, n_fft, n_mels)

    # Apply normalization if requested
    if norm == :slaney do
      enorm = 2.0 / (mel_to_hz(mels[2..-1]) - mel_to_hz(mels[0..-3]))
      enorm_expanded = Nx.new_axis(enorm, 1)
      fbank * enorm_expanded
    else
      fbank
    end
  end

  defnp hz_to_mel(hz) do
    # Convert Hz to mel scale using HTK formula
    1127.0 * Nx.log(1.0 + hz / 700.0)
  end

  defnp mel_to_hz(mel) do
    # Convert mel scale to Hz using HTK formula
    700.0 * (Nx.exp(mel / 1127.0) - 1.0)
  end

  defnp create_triangular_filters(bins, n_fft, n_mels) do
    # Create frequency points tensor
    fft_freqs = Nx.iota({div(n_fft, 2) + 1})

    # Reshape bins for broadcasting
    left_bins = bins[0..-3] |> Nx.reshape({n_mels, 1})
    center_bins = bins[1..-2] |> Nx.reshape({n_mels, 1})
    right_bins = bins[2..-1] |> Nx.reshape({n_mels, 1})

    # Broadcast frequency points for comparison
    freqs = Nx.broadcast(fft_freqs, {n_mels, div(n_fft, 2) + 1})

    # Create left and right slopes
    left_slope = (freqs - left_bins) / (center_bins - left_bins)
    right_slope = (right_bins - freqs) / (right_bins - center_bins)

    # Combine slopes with proper bounds
    left_mask =
      Nx.logical_and(
        freqs >= left_bins,
        freqs <= center_bins
      )

    right_mask =
      Nx.logical_and(
        freqs >= center_bins,
        freqs <= right_bins
      )

    # Create final filterbank
    fbank =
      Nx.select(
        left_mask,
        left_slope,
        Nx.select(
          right_mask,
          right_slope,
          Nx.tensor(0.0)
        )
      )

    # Ensure non-negative values
    Nx.max(fbank, 0.0)
  end
end
