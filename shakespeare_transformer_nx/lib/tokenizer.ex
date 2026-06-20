defmodule ShakespeareTransformer.Tokenizer do
  @moduledoc """
  1
  text = File.read!("priv/input.txt")
  {chars, c2i, i2c} = ShakespeareTransformer.Tokenizer.build_vocab(text)

  IO.puts("Tamanho do vocabulário: \#{length(chars)}")
  IO.inspect(chars)

  encoded = ShakespeareTransformer.Tokenizer.encode("Hello", c2i)
  IO.inspect(encoded)

  decoded = ShakespeareTransformer.Tokenizer.decode(encoded, i2c)
  IO.puts(decoded)
  """

  @doc """
  Constrói vocabulário a partir do texto.
  Retorna {chars, char_to_idx, idx_to_char}
  """
  def build_vocab(text) do
    chars =
      text
      |> String.graphemes()
      |> Enum.uniq()
      |> Enum.sort()

    char_to_idx =
      chars
      |> Enum.with_index()
      |> Map.new()

    idx_to_char =
      chars
      |> Enum.with_index()
      |> Map.new(fn {char, idx} -> {idx, char} end)

    {chars, char_to_idx, idx_to_char}
  end

  @doc "Texto → lista de inteiros"
  def encode(text, char_to_idx) do
    text
    |> String.graphemes()
    |> Enum.map(&Map.fetch!(char_to_idx, &1))
  end

  @doc "Lista de inteiros → texto"
  def decode(indices, idx_to_char) do
    indices
    |> Enum.map(&Map.fetch!(idx_to_char, &1))
    |> Enum.join()
  end
end
