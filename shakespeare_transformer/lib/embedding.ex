defmodule ShakespeareTransformer.Embedding do
  @moduledoc """
  2
  alias ShakespeareTransformer.{Tokenizer, Embedding}

  text = File.read!("priv/input.txt")
  {_chars, c2i, i2c} = Tokenizer.build_vocab(text)

  # hiperparâmetros pequenos pra começar
  vocab_size = map_size(c2i)
  d_model = 32

  # inicializa tabela
  table = Embedding.init(vocab_size, d_model)

  # pega um trecho pequeno
  trecho = "To be or not to be"
  tokens = Tokenizer.encode(trecho, c2i)
  IO.inspect(tokens, label: "tokens")

  # embeddings com posição
  vetores = Embedding.encode_sequence(tokens, table, d_model)

  # mostra o primeiro vetor
  IO.inspect(hd(vetores), label: "embedding do primeiro token")
  IO.puts("dimensão: \#{length(hd(vetores))}")
  """

  @doc """
  Inicializa tabela de embeddings aleatória.
  Retorna lista de {vocab_size} listas de {d_model} floats.
  """
  def init(vocab_size, d_model) do
    for _ <- 1..vocab_size do
      for _ <- 1..d_model do
        # inicialização pequena — importante pra treino estável
        (:rand.uniform() - 0.5) * 0.1
      end
    end
  end

  @doc """
  Busca embedding de um token pelo índice.
  """
  def lookup(table, idx) do
    Enum.at(table, idx)
  end

  @doc """
  Gera vetor de posição pra uma posição e d_model.
  """
  def positional_encoding(pos, d_model) do
    for dim <- 0..(d_model - 1) do
      angle = pos / :math.pow(10000, dim / d_model)

      if rem(dim, 2) == 0 do
        :math.sin(angle)
      else
        :math.cos(angle)
      end
    end
  end

  @doc """
  Soma dois vetores elemento a elemento.
  """
  def add(v1, v2) do
    Enum.zip_with(v1, v2, &(&1 + &2))
  end

  @doc """
  Processa sequência inteira de tokens.
  Retorna lista de vetores {token + posição}.
  """
  def encode_sequence(tokens, table, d_model) do
    tokens
    |> Enum.with_index()
    |> Enum.map(fn {token_idx, pos} ->
      embedding = lookup(table, token_idx)
      pos_enc = positional_encoding(pos, d_model)
      add(embedding, pos_enc)
    end)
  end
end
