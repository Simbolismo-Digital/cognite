defmodule ShakespeareTransformer.LanguageHead do
  @moduledoc """
  6
  alias ShakespeareTransformer.{Tokenizer, Embedding, Attention, FeedForward, TransformerBlock, LanguageHead}

  text       = File.read!("priv/input.txt")
  {_chars, c2i, i2c} = Tokenizer.build_vocab(text)

  vocab_size = map_size(c2i)
  d_model    = 32
  n_heads    = 2
  d_head     = 16

  table      = Embedding.init(vocab_size, d_model)
  params     = TransformerBlock.init(d_model, n_heads, d_head)
  params2    = TransformerBlock.init(d_model, n_heads, d_head)
  w_head     = LanguageHead.init(d_model, vocab_size)

  tokens  = Tokenizer.encode("To be or not", c2i)
  targets = Enum.drop(tokens, 1)
  inputs  = Enum.drop(tokens, -1)

  vetores_treino = Embedding.encode_sequence(inputs, table, d_model)
  block_out      = TransformerBlock.forward(vetores_treino, params)
  block_out2     = TransformerBlock.forward(block_out, params2)

  loss = LanguageHead.sequence_loss(block_out2, targets, w_head)
  IO.puts("loss inicial: \#{loss}")
  """
  @doc """
  Projeta vetor [d_model] → [vocab_size].
  Retorna logits — scores antes do softmax.
  """
  def project(vec, w_head) do
    w_head
    |> transpose()
    |> Enum.map(fn col -> dot_product(vec, col) end)
  end

  @doc "Softmax — transforma logits em probabilidades"
  def softmax(logits) do
    max_l = Enum.max(logits)
    exps  = Enum.map(logits, fn x -> :math.exp(x - max_l) end)
    sum   = Enum.sum(exps)
    Enum.map(exps, &(&1 / sum))
  end

  @doc """
  Cross-entropy loss pra um passo.
  probs:      lista de probabilidades [vocab_size]
  target_idx: índice do token correto
  """
  def cross_entropy(probs, target_idx) do
    p = Enum.at(probs, target_idx)
    -:math.log(p + 1.0e-8)
  end

  @doc """
  Loss média sobre toda a sequência.
  outputs:  lista de vetores [d_model] — output dos blocos
  targets:  lista de índices — próximo token em cada posição
  w_head:   matriz de projeção [d_model × vocab_size]
  """
  def sequence_loss(outputs, targets, w_head) do
    outputs
    |> Enum.zip(targets)
    |> Enum.map(fn {vec, target} ->
      vec
      |> project(w_head)
      |> softmax()
      |> cross_entropy(target)
    end)
    |> then(fn losses ->
      Enum.sum(losses) / length(losses)
    end)
  end

  @doc "Inicializa matriz de projeção [d_model × vocab_size]"
  def init(d_model, vocab_size) do
    scale = :math.sqrt(2.0 / d_model)

    for _ <- 1..d_model do
      for _ <- 1..vocab_size do
        (:rand.uniform() - 0.5) * scale
      end
    end
  end

  defp dot_product(v1, v2) do
    Enum.zip_with(v1, v2, &(&1 * &2))
    |> Enum.sum()
  end

  defp transpose(matrix) do
    matrix
    |> Enum.zip()
    |> Enum.map(&Tuple.to_list/1)
  end
end
