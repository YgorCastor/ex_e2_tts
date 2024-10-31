defmodule ExE2Tts.Config do
  @moduledoc """
  Configuration settings for E2/F5 TTS model.
  """

  # Audio settings
  def target_sample_rate, do: 24_000
  def n_mel_channels, do: 100
  def hop_length, do: 256
  def target_rms, do: 0.1

  # Model architecture settings
  def model_config(:f5_tts_base) do
    %{
      dim: 1024,
      depth: 22,
      heads: 16,
      ff_mult: 2,
      text_dim: 512,
      conv_layers: 4
    }
  end

  def model_config(:e2_tts_base) do
    %{
      dim: 1024,
      depth: 24,
      heads: 16,
      ff_mult: 4
    }
  end

  # Training settings
  def training_config do
    %{
      batch_size: 32,
      learning_rate: 1.0e-4,
      max_epochs: 100,
      warmup_steps: 1000
    }
  end

  # Inference settings
  def inference_config do
    %{
      nfe_step: 32,
      cfg_strength: 2.0,
      ode_method: :euler,
      sway_sampling_coef: -1.0,
      speed: 1.0
    }
  end
end
