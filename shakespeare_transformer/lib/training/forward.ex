defmodule ShakespeareTransformer.Training.Forward do
  @moduledoc """
  Forward pass com cache — guarda intermediários necessários pro backward.

  Cada função retorna {output, cache} onde cache contém
  tudo que backward precisa além dos pesos.
  """

  alias ShakespeareTransformer.{Embedding, Attention, FeedForward, TransformerBlock, LanguageHead, Training}

  # ---------------------------------------------------------------------------
  # Attention com cache
  # ---------------------------------------------------------------------------

  @doc """
  Self-attention com cache.

  Retorna {output, cache} onde cache = %{
    queries, keys, values, weights
  }
  """
  def self_attention(inputs, w_q, w_k, w_v) do
    d_head      = length(hd(Training.transpose(w_q)))
    scale       = 1.0 / :math.sqrt(d_head)

    queries = Enum.map(inputs, &Attention.vec_mat_mul(&1, w_q))
    keys    = Enum.map(inputs, &Attention.vec_mat_mul(&1, w_k))
    values  = Enum.map(inputs, &Attention.vec_mat_mul(&1, w_v))

    output =
      Enum.map(queries, fn q ->
        scores =
          Enum.map(keys, fn k ->
            Training.dot_product(q, k) * scale
          end)

        weights = Attention.softmax(scores)
        Attention.weighted_sum(weights, values)
      end)

    weights_matrix =
      Enum.map(queries, fn q ->
        scores = Enum.map(keys, fn k -> Training.dot_product(q, k) * scale end)
        Attention.softmax(scores)
      end)

    cache = %{queries: queries, keys: keys, values: values, weights: weights_matrix}
    {output, cache}
  end

  @doc """
  Multi-head attention com cache.

  Retorna {output, cache} onde cache = %{
    head_caches:  cache de cada cabeça
    head_outputs: output de cada cabeça [n_heads × seq × d_head]
    concat_list:  vetores concatenados [seq × (n_heads*d_head)]
  }
  """
  def multi_head_attention(inputs, heads, w_out) do
    {head_outputs, head_caches} =
      heads
      |> Enum.map(fn {w_q, w_k, w_v} ->
        self_attention(inputs, w_q, w_k, w_v)
      end)
      |> Enum.unzip()

    # concatena outputs de todas as cabeças por posição
    concat_list =
      head_outputs
      |> Enum.zip_with(fn vectors -> Enum.concat(vectors) end)

    # projeção final
    output =
      Enum.map(concat_list, fn c ->
        Attention.vec_mat_mul(c, w_out)
      end)

    cache = %{
      head_caches:  head_caches,
      head_outputs: head_outputs,
      concat_list:  concat_list
    }

    {output, cache}
  end

  # ---------------------------------------------------------------------------
  # Feed-forward com cache
  # ---------------------------------------------------------------------------

  @doc """
  Feed-forward com cache.

  Retorna {output, cache} onde cache = %{h, a}
    h: output da primeira linear (antes do ReLU)
    a: output do ReLU
  """
  def feed_forward(x, w1, b1, w2, b2) do
    h = Training.linear_forward(x, w1, b1)
    a = FeedForward.relu(h)
    y = Training.linear_forward(a, w2, b2)
    {y, %{h: h, a: a}}
  end

  @doc "Feed-forward sobre sequência inteira com cache por posição"
  def feed_forward_sequence(inputs, w1, b1, w2, b2) do
    {outputs, caches} =
      inputs
      |> Enum.map(&feed_forward(&1, w1, b1, w2, b2))
      |> Enum.unzip()

    hs = Enum.map(caches, & &1.h)
    as = Enum.map(caches, & &1.a)

    {outputs, %{h_list: hs, a_list: as}}
  end

  # ---------------------------------------------------------------------------
  # Transformer block com cache
  # ---------------------------------------------------------------------------

  @doc """
  Forward de um transformer block com cache completo.

  Retorna {output, cache} onde cache = %{
    inputs, normed1, attn_out, x1,
    normed2, ff_h, ff_a,
    attn_cache
  }
  """
  def transformer_block(inputs, params) do
    # sublayer 1 — multi-head attention + residual
    normed1 = Enum.map(inputs, &TransformerBlock.layer_norm/1)

    {attn_out, attn_cache} =
      multi_head_attention(normed1, params.heads, params.w_out)

    x1 = Enum.zip_with(inputs, attn_out, &Training.add_vec/2)

    # sublayer 2 — feed-forward + residual
    normed2 = Enum.map(x1, &TransformerBlock.layer_norm/1)

    {ff_out, ff_cache} =
      feed_forward_sequence(normed2, params.w1, params.b1, params.w2, params.b2)

    x2 = Enum.zip_with(x1, ff_out, &Training.add_vec/2)

    cache = %{
      inputs:     inputs,
      normed1:    normed1,
      attn_out:   attn_out,
      attn_cache: attn_cache,
      x1:         x1,
      normed2:    normed2,
      ff_h:       ff_cache.h_list,
      ff_a:       ff_cache.a_list
    }

    {x2, cache}
  end

  # ---------------------------------------------------------------------------
  # Forward completo do modelo
  # ---------------------------------------------------------------------------

  @doc """
  Forward pass completo com cache.

  tokens:      lista de índices (input)
  targets:     lista de índices (próximo token em cada posição)
  model:       %{table, blocks_params, w_head, d_model}

  Retorna {loss, cache} onde cache contém tudo pra backward.
  """
  def forward(tokens, targets, model) do
    # embedding + positional encoding
    vetores = Embedding.encode_sequence(tokens, model.table, model.d_model)

    # passa pelos blocos transformer
    {final_output, block_caches} =
      Enum.reduce(model.blocks_params, {vetores, []}, fn params, {input, caches} ->
        {output, cache} = transformer_block(input, params)
        {output, caches ++ [cache]}
      end)

    # language head — projeta pra vocabulário e calcula loss
    {loss, lh_cache} = language_head_loss(final_output, targets, model.w_head)

    cache = %{
      tokens:       tokens,
      vetores:      vetores,
      block_caches: block_caches,
      final_output: final_output,
      lh_cache:     lh_cache
    }

    {loss, cache}
  end

  # ---------------------------------------------------------------------------
  # Language head com cache
  # ---------------------------------------------------------------------------

  @doc """
  Projeção final + loss com cache.

  Retorna {loss, cache} onde cache = %{
    logits_list, probs_list, targets
  }
  """
  def language_head_loss(outputs, targets, w_head) do
    logits_list = Enum.map(outputs, &LanguageHead.project(&1, w_head))
    probs_list  = Enum.map(logits_list, &LanguageHead.softmax/1)

    losses =
      Enum.zip(probs_list, targets)
      |> Enum.map(fn {probs, target} ->
        LanguageHead.cross_entropy(probs, target)
      end)

    loss = Enum.sum(losses) / length(losses)

    cache = %{
      logits_list: logits_list,
      probs_list:  probs_list,
      targets:     targets
    }

    {loss, cache}
  end
end
