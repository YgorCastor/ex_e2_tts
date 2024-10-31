defmodule ExE2Tts.MixProject do
  use Mix.Project

  def project do
    [
      app: :ex_e2_tts,
      version: "0.1.0",
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      # Deep Learning
      {:nx, "~> 0.9"},
      {:exla, "~> 0.9"},
      {:axon, "~> 0.7"},
      {:torchx, "~> 0.9"},
      # Utilities
      {:jason, "~> 1.4"},
      {:flow, "~> 1.2"},
      # HTTP client
      {:req, "~> 0.5"},

      # Development and testing
      {:benchee, "~> 1.1", only: :dev},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:dialyxir, "~> 1.3", only: [:dev], runtime: false},
      {:ex_doc, "~> 0.30.3", only: :dev, runtime: false}
    ]
  end
end
