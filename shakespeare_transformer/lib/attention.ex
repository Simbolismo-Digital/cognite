defmodule ShakespeareTransformer.Attention do
  @moduledoc """
  3
  alias ShakespeareTransformer.{Tokenizer, Embedding, Attention}

  text = File.read!("priv/input.txt")
  {_chars, c2i, _i2c} = Tokenizer.build_vocab(text)

  vocab_size = map_size(c2i)
  d_model = 32
  d_head  = 16

  # embedding
  table   = Embedding.init(vocab_size, d_model)
  tokens  = Tokenizer.encode("To be or not", c2i)
  vetores = Embedding.encode_sequence(tokens, table, d_model)

  # pesos de atenção
  w_q = Attention.init_weights(d_model, d_head)
  w_k = Attention.init_weights(d_model, d_head)
  w_v = Attention.init_weights(d_model, d_head)

  # self-attention
  output = Attention.self_attention(vetores, w_q, w_k, w_v)

  IO.puts("entrada: \#{length(vetores)} vetores de \#{length(hd(vetores))} dims")
  IO.puts("saída:   \#{length(output)} vetores de \#{length(hd(output))} dims")
  IO.inspect(hd(output), label: "primeiro vetor de saída")

  # multi-head-attention
  n_heads = 2
  d_head  = 16   # n_heads * d_head == d_model == 32

  heads = Attention.init_heads(n_heads, d_model, d_head)
  w_out = Attention.init_weights(n_heads * d_head, d_model)

  output = Attention.multi_head_attention(vetores, heads, w_out)

  IO.puts("entrada: \#{length(vetores)} vetores de \#{length(hd(vetores))} dims")
  IO.puts("saída:   \#{length(output)} vetores de \#{length(hd(output))} dims")
  """

  @doc "Produto escalar entre dois vetores"
  def dot_product(v1, v2) do
    Enum.zip_with(v1, v2, &(&1 * &2))
    |> Enum.sum()
  end

  @doc "Multiplica vetor [d_model] por matriz [d_model × d_head] → [d_head]"
  def vec_mat_mul(vec, matrix) do
    # transpõe a matriz pra percorrer colunas como linhas
    matrix
    |> transpose()
    |> Enum.map(fn col -> dot_product(vec, col) end)
  end

  @doc "Transpõe matriz (lista de listas)"
  def transpose(matrix) do
    matrix
    |> Enum.zip()
    |> Enum.map(&Tuple.to_list/1)
  end

  @doc "Softmax de uma lista de floats"
  def softmax(scores) do
    max_score = Enum.max(scores)

    exps = Enum.map(scores, fn s ->
      :math.exp(s - max_score)  # subtrai max pra estabilidade numérica
    end)

    sum = Enum.sum(exps)
    Enum.map(exps, &(&1 / sum))
  end

  @doc "Escala vetor por escalar"
  def scale(vec, scalar) do
    Enum.map(vec, &(&1 * scalar))
  end

  @doc "Soma lista de vetores ponderados"
  def weighted_sum(weights, values) do
    Enum.zip(weights, values)
    |> Enum.map(fn {w, v} -> scale(v, w) end)
    |> Enum.reduce(fn v, acc ->
      Enum.zip_with(v, acc, &(&1 + &2))
    end)
  end

  @doc """
  Self-attention completo.

  inputs: lista de vetores [d_model]
  w_q, w_k, w_v: matrizes [d_model × d_head]

  retorna: lista de vetores enriquecidos
  """
  def self_attention(inputs, w_q, w_k, w_v) do
    d_head = length(hd(w_q))
    scale_factor = 1.0 / :math.sqrt(d_head)

    # projeta cada input em Q, K, V
    queries = Enum.map(inputs, &vec_mat_mul(&1, w_q))
    keys    = Enum.map(inputs, &vec_mat_mul(&1, w_k))
    values  = Enum.map(inputs, &vec_mat_mul(&1, w_v))

    # pra cada posição, calcula atenção sobre toda a sequência
    Enum.map(queries, fn q ->
      # score contra todas as keys
      scores =
        Enum.map(keys, fn k ->
          dot_product(q, k) * scale_factor
        end)

      # softmax dos scores
      weights = softmax(scores)

      # soma ponderada dos values
      weighted_sum(weights, values)
    end)
  end

  @doc "Inicializa matriz de pesos aleatória [d_model × d_head]"
  def init_weights(d_model, d_head) do
    scale = :math.sqrt(2.0 / (d_model + d_head))

    for _ <- 1..d_model do
      for _ <- 1..d_head do
        (:rand.uniform() - 0.5) * scale
      end
    end
  end

  @doc """
  Multi-head attention.

  inputs:  lista de vetores [d_model]
  heads:   lista de {w_q, w_k, w_v} — um por cabeça
  w_out:   matriz de projeção final [n_heads*d_head × d_model]

  retorna: lista de vetores [d_model]
  """
  def multi_head_attention(inputs, heads, w_out) do
    # roda cada cabeça independentemente
    head_outputs =
      Enum.map(heads, fn {w_q, w_k, w_v} ->
        self_attention(inputs, w_q, w_k, w_v)
      end)

    # concatena outputs de todas as cabeças pra cada posição
    Enum.zip_with(head_outputs, fn vectors ->
      # vectors é lista com o output de cada cabeça pra essa posição
      concatenated = Enum.concat(vectors)

      # projeta de volta pra d_model
      vec_mat_mul(concatenated, w_out)
    end)
  end

  @doc "Inicializa N cabeças de atenção"
  def init_heads(n_heads, d_model, d_head) do
    for _ <- 1..n_heads do
      {
        init_weights(d_model, d_head),
        init_weights(d_model, d_head),
        init_weights(d_model, d_head)
      }
    end
  end
end
