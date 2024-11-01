defmodule ExE2Tts.Dataset.Cache do
  @moduledoc """
  Caching mechanism for preprocessed mel spectrograms.
  Supports both memory and disk caching with automatic invalidation.
  """

  require Logger

  defstruct [
    :cache_dir,
    :memory_cache,
    :cache_type,
    :max_memory_items,
    :stats
  ]

  @type cache_type :: :memory | :disk | :both
  @type stats :: %{
          hits: integer(),
          misses: integer(),
          memory_size: integer(),
          disk_size: integer()
        }

  @type t :: %__MODULE__{
          cache_dir: String.t() | nil,
          memory_cache: :ets.tid() | nil,
          cache_type: cache_type(),
          max_memory_items: pos_integer() | nil,
          stats: stats()
        }

  @doc """
  Creates a new cache instance.

  ## Options
    * :cache_type - :memory | :disk | :both (default: :memory)
    * :cache_dir - Directory for disk cache (required for :disk and :both)
    * :max_memory_items - Maximum items to keep in memory (default: 10000)
  """
  def new(opts \\ []) do
    cache_type = opts[:cache_type] || :memory
    cache_dir = opts[:cache_dir]
    max_memory_items = opts[:max_memory_items] || 10_000

    # Validate options
    if cache_type in [:disk, :both] and not is_binary(cache_dir) do
      raise ArgumentError, "cache_dir is required for disk caching"
    end

    # Create cache directory if needed
    if cache_type in [:disk, :both] do
      File.mkdir_p!(cache_dir)
    end

    # Initialize memory cache if needed
    memory_cache =
      if cache_type in [:memory, :both] do
        :ets.new(:mel_spec_cache, [:set, :public])
      end

    %__MODULE__{
      cache_dir: cache_dir,
      memory_cache: memory_cache,
      cache_type: cache_type,
      max_memory_items: max_memory_items,
      stats: %{hits: 0, misses: 0, memory_size: 0, disk_size: 0}
    }
  end

  @doc """
  Gets a cached mel spectrogram or generates and caches it.
  """
  def get_or_store(cache, key, generator_fn) do
    case get(cache, key) do
      {:ok, value} ->
        update_stats(cache, :hits)
        {:ok, value}

      :error ->
        update_stats(cache, :misses)
        value = generator_fn.()
        {:ok, _} = store(cache, key, value)
        {:ok, value}
    end
  end

  @doc """
  Stores a value in the cache.
  """
  def store(%__MODULE__{} = cache, key, value) do
    case cache.cache_type do
      :memory ->
        store_in_memory(cache, key, value)

      :disk ->
        store_on_disk(cache, key, value)

      :both ->
        with {:ok, _} <- store_in_memory(cache, key, value),
             {:ok, _} <- store_on_disk(cache, key, value) do
          {:ok, value}
        end
    end
  end

  @doc """
  Retrieves a value from the cache.
  """
  def get(%__MODULE__{} = cache, key) do
    case cache.cache_type do
      :memory ->
        get_from_memory(cache, key)

      :disk ->
        get_from_disk(cache, key)

      :both ->
        case get_from_memory(cache, key) do
          {:ok, value} -> {:ok, value}
          :error -> get_from_disk(cache, key)
        end
    end
  end

  @doc """
  Clears the cache.
  """
  def clear(%__MODULE__{} = cache) do
    if cache.memory_cache do
      :ets.delete_all_objects(cache.memory_cache)
    end

    if cache.cache_dir do
      File.rm_rf!(cache.cache_dir)
      File.mkdir_p!(cache.cache_dir)
    end

    %{cache | stats: %{hits: 0, misses: 0, memory_size: 0, disk_size: 0}}
  end

  defp store_in_memory(%{memory_cache: nil}, _key, _value), do: {:error, :no_memory_cache}

  defp store_in_memory(%{memory_cache: cache} = state, key, value) do
    # Manage cache size
    if :ets.info(cache, :size) >= state.max_memory_items do
      # Remove random item if at capacity
      [{old_key, _}] = :ets.tab2list(cache) |> Enum.take_random(1)
      :ets.delete(cache, old_key)
    end

    true = :ets.insert(cache, {key, value})
    {:ok, value}
  end

  defp store_on_disk(%{cache_dir: nil}, _key, _value), do: {:error, :no_disk_cache}

  defp store_on_disk(%{cache_dir: dir}, key, value) do
    path = cache_path(dir, key)
    binary = Nx.serialize(value)
    File.write!(path, binary)
    {:ok, value}
  end

  defp get_from_memory(%{memory_cache: nil}, _key), do: :error

  defp get_from_memory(%{memory_cache: cache}, key) do
    case :ets.lookup(cache, key) do
      [{^key, value}] -> {:ok, value}
      [] -> :error
    end
  end

  defp get_from_disk(%{cache_dir: nil}, _key), do: :error

  defp get_from_disk(%{cache_dir: dir}, key) do
    path = cache_path(dir, key)

    if File.exists?(path) do
      value =
        path
        |> File.read!()
        |> Nx.deserialize()

      {:ok, value}
    else
      :error
    end
  end

  defp cache_path(dir, key) do
    filename = :crypto.hash(:sha256, :erlang.term_to_binary(key)) |> Base.encode16(case: :lower)
    Path.join(dir, filename)
  end

  defp update_stats(cache, :hits) do
    put_in(cache.stats.hits, cache.stats.hits + 1)
  end

  defp update_stats(cache, :misses) do
    put_in(cache.stats.misses, cache.stats.misses + 1)
  end
end
