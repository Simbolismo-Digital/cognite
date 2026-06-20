defmodule ShakespeareTransformer.NxModel do
  @moduledoc """
  Mesmo transformer que você implementou na mão — agora em Axon.

  Arquitetura idêntica à versão Elixir puro:
    embedding + positional encoding
    -> N transformer blocks (multi-head attention + feed-forward, com residual + layer norm)
    -> language head (projeção pro vocabulário)

  A diferença é só o substrato: Nx.Defn compila as operações,
  Axon cuida do autograd. Mesma matemática que você derivou na mão.
  """

  @doc """
  Constrói o grafo do modelo.

  opts:
    vocab_size, d_model, n_heads, n_blocks, seq_len
  """
  def build(opts) do
    vocab_size = Keyword.fetch!(opts, :vocab_size)
    d_model    = Keyword.fetch!(opts, :d_model)
    n_heads    = Keyword.fetch!(opts, :n_heads)
    n_blocks   = Keyword.fetch!(opts, :n_blocks)
    seq_len    = Keyword.fetch!(opts, :seq_len)
    batch_size = Keyword.get(opts, :batch_size, nil)

    input = Axon.input("tokens", shape: {batch_size, seq_len})

    # --- embedding + positional encoding ---
    x =
      input
      |> Axon.embedding(vocab_size, d_model, name: "token_embedding")
      |> add_positional_encoding(seq_len, d_model)

    # --- N transformer blocks ---
    x =
      Enum.reduce(1..n_blocks, x, fn i, acc ->
        transformer_block(acc, d_model, n_heads, seq_len, "block_#{i}")
      end)

    # --- language head ---
    x
    |> Axon.dense(vocab_size, name: "language_head")
  end

  # ---------------------------------------------------------------------------
  # Positional encoding — mesma fórmula seno/cosseno do paper
  # ---------------------------------------------------------------------------

  defp add_positional_encoding(x, seq_len, d_model) do
    pe = positional_encoding_matrix(seq_len, d_model)

    Axon.layer(
      fn input, pe_tensor, _opts ->
        Nx.add(input, pe_tensor)
      end,
      [x, Axon.constant(pe)],
      name: "positional_encoding"
    )
  end

  defp positional_encoding_matrix(seq_len, d_model) do
    positions = Nx.iota({seq_len, 1}, type: :f32)
    dims      = Nx.iota({1, d_model}, type: :f32)

    angle_rates =
      Nx.pow(10_000.0, Nx.divide(dims, d_model))

    angles = Nx.divide(positions, angle_rates)

    # pares: seno, ímpares: cosseno
    even_mask =
      Nx.iota({1, d_model})
      |> Nx.remainder(2)
      |> Nx.equal(0)
      |> Nx.broadcast({seq_len, d_model})

    sines   = Nx.sin(angles)
    cosines = Nx.cos(angles)

    Nx.select(even_mask, sines, cosines)
  end

  # ---------------------------------------------------------------------------
  # Transformer block — attention + feed-forward + residual + layer norm
  # ---------------------------------------------------------------------------

  defp transformer_block(x, d_model, n_heads, seq_len, name) do
    # sublayer 1: multi-head attention + residual
    normed1 = Axon.layer_norm(x, name: "#{name}_ln1")
    attn    = multi_head_attention(normed1, d_model, n_heads, seq_len, "#{name}_attn")
    attn    = Axon.dropout(attn, rate: 0.1, name: "#{name}_drop1")
    x1      = Axon.add(x, attn, name: "#{name}_residual1")

    # sublayer 2: feed-forward + residual
    normed2 = Axon.layer_norm(x1, name: "#{name}_ln2")
    ff      = feed_forward(normed2, d_model, "#{name}_ff")
    ff      = Axon.dropout(ff, rate: 0.1, name: "#{name}_drop2")
    Axon.add(x1, ff, name: "#{name}_residual2")
  end

  # ---------------------------------------------------------------------------
  # Multi-head self-attention
  # ---------------------------------------------------------------------------

  defp multi_head_attention(x, d_model, n_heads, seq_len, name) do
    d_head = div(d_model, n_heads)

    q = Axon.dense(x, d_model, name: "#{name}_q", use_bias: false)
    k = Axon.dense(x, d_model, name: "#{name}_k", use_bias: false)
    v = Axon.dense(x, d_model, name: "#{name}_v", use_bias: false)

    attended =
      Axon.layer(
        &scaled_dot_product_attention/4,
        [q, k, v],
        name: "#{name}_sdpa",
        n_heads: n_heads,
        d_head: d_head,
        seq_len: seq_len
      )

    Axon.dense(attended, d_model, name: "#{name}_out")
  end

  defp scaled_dot_product_attention(q, k, v, opts) do
    n_heads = opts[:n_heads]
    d_head  = opts[:d_head]

    {batch, seq_len, d_model} = Nx.shape(q)

    # reshape pra [batch, n_heads, seq_len, d_head]
    reshape_heads = fn t ->
      t
      |> Nx.reshape({batch, seq_len, n_heads, d_head})
      |> Nx.transpose(axes: [0, 2, 1, 3])
    end

    qh = reshape_heads.(q)
    kh = reshape_heads.(k)
    vh = reshape_heads.(v)

    scale = 1.0 / :math.sqrt(d_head)

    # scores: [batch, n_heads, seq_len, seq_len]
    scores =
      Nx.dot(qh, [3], [0, 1], kh, [3], [0, 1])
      |> Nx.multiply(scale)

    # causal mask — impede atenção em posições futuras (j > i)
    row_idx = Nx.iota({seq_len, 1})
    col_idx = Nx.iota({1, seq_len})

    causal_mask =
      Nx.greater_equal(row_idx, col_idx)
      |> Nx.reshape({1, 1, seq_len, seq_len})
      |> Nx.broadcast(Nx.shape(scores))

    neg_inf = Nx.Constants.neg_infinity(Nx.type(scores))
    masked_scores = Nx.select(causal_mask, scores, neg_inf)

    weights = Axon.Activations.softmax(masked_scores, axis: -1)

    # output: [batch, n_heads, seq_len, d_head]
    out = Nx.dot(weights, [3], [0, 1], vh, [2], [0, 1])

    # volta pra [batch, seq_len, d_model]
    out
    |> Nx.transpose(axes: [0, 2, 1, 3])
    |> Nx.reshape({batch, seq_len, d_model})
  end

  # ---------------------------------------------------------------------------
  # Feed-forward
  # ---------------------------------------------------------------------------

  defp feed_forward(x, d_model, name) do
    d_ff = d_model * 4

    x
    |> Axon.dense(d_ff, activation: :relu, name: "#{name}_1")
    |> Axon.dense(d_model, name: "#{name}_2")
  end
end
