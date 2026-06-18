defmodule ShakespeareTransformer.Training.Backward do
  @moduledoc """
  Backward pass completo — propaga gradientes do language head
  até o embedding, atravessando todos os blocos transformer.
  """

  alias ShakespeareTransformer.Training.Gradients
  alias ShakespeareTransformer.Training

  # ---------------------------------------------------------------------------
  # Language head backward
  # ---------------------------------------------------------------------------

  @doc """
  Backward do language head.

  Retorna {d_outputs, dw_head}
    d_outputs: gradiente em relação ao output dos blocos [seq × d_model]
    dw_head:   gradiente em relação à matriz de projeção
  """
  def language_head_backward(final_output, lh_cache, w_head) do
    %{probs_list: probs_list, targets: targets} = lh_cache

    n = length(targets)

    dlogits_list =
      Enum.zip(probs_list, targets)
      |> Enum.map(fn {probs, target} ->
        grad = Gradients.softmax_cross_entropy_backward(probs, target)
        Training.scale_vec(grad, 1.0 / n)
      end)

    dw_head =
      Enum.zip(final_output, dlogits_list)
      |> Enum.map(fn {out, dlogits} ->
        Training.outer_product(out, dlogits)
      end)
      |> Enum.reduce(&Training.add_mat/2)

    d_outputs =
      Enum.map(dlogits_list, fn dlogits ->
        Training.linear_backward_input(dlogits, w_head)
      end)

    {d_outputs, dw_head}
  end

  # ---------------------------------------------------------------------------
  # Transformer block backward
  # ---------------------------------------------------------------------------

  @doc """
  Backward de um transformer block.

  Retorna {dx_inputs, dparams}
  """
  def transformer_block_backward(cache, params, dy_list) do
    # --- sublayer 2: feed-forward + residual ---
    {dx_ff_list, dw1, db1, dw2, db2} =
      Enum.zip_with(
        [cache.normed2, cache.ff_h, cache.ff_a, dy_list],
        fn [n2, h, a, dy] ->
          Training.feed_forward_backward(n2, h, a, dy, params.w1, params.b1, params.w2, params.b2)
        end
      )
      |> unzip5()

    dw1 = Enum.reduce(dw1, &Training.add_mat/2)
    db1 = Enum.reduce(db1, &Training.add_vec/2)
    dw2 = Enum.reduce(dw2, &Training.add_mat/2)
    db2 = Enum.reduce(db2, &Training.add_vec/2)

    # gradiente pelo layer norm 2
    dx_normed2 =
      Enum.zip_with(cache.x1, dx_ff_list, fn x1, dx_ff ->
        Gradients.layer_norm_backward(x1, dx_ff)
      end)

    # residual: soma gradiente direto + pelo ff
    dx1 = Enum.zip_with(dy_list, dx_normed2, &Training.add_vec/2)

    # --- sublayer 1: multi-head attention + residual ---
    {dx_attn_list, dheads, dw_out} =
      multi_head_backward(cache.normed1, cache.attn_cache, params.heads, params.w_out, dx1)

    # gradiente pelo layer norm 1
    dx_normed1 =
      Enum.zip_with(cache.inputs, dx_attn_list, fn inp, dx_a ->
        Gradients.layer_norm_backward(inp, dx_a)
      end)

    # residual
    dx_inputs = Enum.zip_with(dx1, dx_normed1, &Training.add_vec/2)

    dparams = %{
      heads: dheads,
      w_out: dw_out,
      w1: dw1, b1: db1,
      w2: dw2, b2: db2
    }

    {dx_inputs, dparams}
  end

  # ---------------------------------------------------------------------------
  # Multi-head attention backward
  # ---------------------------------------------------------------------------

  defp multi_head_backward(inputs, attn_cache, heads, w_out, dy_list) do
    # gradiente da projeção final
    dw_out =
      Enum.zip_with(attn_cache.concat_list, dy_list, fn c, dy ->
        Training.outer_product(c, dy)
      end)
      |> Enum.reduce(&Training.add_mat/2)

    dconcat_list =
      Enum.map(dy_list, fn dy ->
        Training.linear_backward_input(dy, w_out)
      end)

    # divide gradiente concatenado por cabeça
    d_head =
      attn_cache.head_outputs
      |> hd()
      |> hd()
      |> length()

    dhead_outputs_per_pos =
      Enum.map(dconcat_list, fn dc ->
        Enum.chunk_every(dc, d_head)
      end)

    # transpõe: [seq × n_heads × d_head] → [n_heads × seq × d_head]
    n_heads = length(heads)

    dhead_outputs =
      Enum.map(0..(n_heads - 1), fn h_idx ->
        Enum.map(dhead_outputs_per_pos, fn pos -> Enum.at(pos, h_idx) end)
      end)

    # backward de cada cabeça
    {dheads, dx_per_head} =
      Enum.zip_with(
        [heads, attn_cache.head_caches, dhead_outputs],
        fn [{w_q, w_k, w_v}, hc, dy_head] ->
          {dx, dw_q, dw_k, dw_v} =
            attention_backward(inputs, hc, w_q, w_k, w_v, dy_head)
          {{dw_q, dw_k, dw_v}, dx}
        end
      )
      |> Enum.unzip()

    # soma contribuições de cada cabeça
    dx_list =
      dx_per_head
      |> Enum.reduce(fn dx, acc ->
        Enum.zip_with(dx, acc, &Training.add_vec/2)
      end)

    {dx_list, dheads, dw_out}
  end

  # ---------------------------------------------------------------------------
  # Self-attention backward
  # ---------------------------------------------------------------------------

  defp attention_backward(inputs, cache, w_q, w_k, w_v, dy_list) do
    %{queries: queries, keys: keys, values: values, weights: weights} = cache

    scale  = 1.0 / :math.sqrt(length(hd(queries)))
    d_head = length(hd(values))

    # gradiente dos values
    # output[i] = sum_j(weights[i][j] * values[j])
    # dvalues[j] = sum_i(weights[i][j] * dy[i])
    weights_t = Training.transpose(weights)

    dvalues =
      Enum.map(weights_t, fn col ->
        Enum.zip(col, dy_list)
        |> Enum.reduce(Training.zeros(d_head), fn {w, dy}, acc ->
          Training.add_vec(acc, Training.scale_vec(dy, w))
        end)
      end)

    # gradiente dos weights (antes do softmax)
    dweights =
      Enum.map(dy_list, fn dyi ->
        Enum.map(values, fn vj ->
          Training.dot_product(dyi, vj)
        end)
      end)

    # gradiente dos scores (passa pelo softmax backward + scale)
    dscores =
      Enum.zip_with(dweights, weights, fn dw_row, w_row ->
        Training.softmax_backward(w_row, dw_row)
        |> Training.scale_vec(scale)
      end)

    # gradiente de Q e K
    dqueries =
      Enum.map(dscores, fn ds_row ->
        Enum.zip(ds_row, keys)
        |> Enum.reduce(Training.zeros(d_head), fn {ds, k}, acc ->
          Training.add_vec(acc, Training.scale_vec(k, ds))
        end)
      end)

    dscores_t = Training.transpose(dscores)

    dkeys =
      Enum.map(dscores_t, fn ds_col ->
        Enum.zip(ds_col, queries)
        |> Enum.reduce(Training.zeros(d_head), fn {ds, q}, acc ->
          Training.add_vec(acc, Training.scale_vec(q, ds))
        end)
      end)

    # gradiente das matrizes de projeção
    dw_q =
      Enum.zip_with(inputs, dqueries, fn x, dq ->
        Training.outer_product(x, dq)
      end)
      |> Enum.reduce(&Training.add_mat/2)

    dw_k =
      Enum.zip_with(inputs, dkeys, fn x, dk ->
        Training.outer_product(x, dk)
      end)
      |> Enum.reduce(&Training.add_mat/2)

    dw_v =
      Enum.zip_with(inputs, dvalues, fn x, dv ->
        Training.outer_product(x, dv)
      end)
      |> Enum.reduce(&Training.add_mat/2)

    # gradiente dos inputs — três caminhos
    dx_list =
      Enum.zip_with([dqueries, dkeys, dvalues], fn [dq, dk, dv] ->
        Training.linear_backward_input(dq, w_q)
        |> Training.add_vec(Training.linear_backward_input(dk, w_k))
        |> Training.add_vec(Training.linear_backward_input(dv, w_v))
      end)

    {dx_list, dw_q, dw_k, dw_v}
  end

  # ---------------------------------------------------------------------------
  # Backward completo do modelo
  # ---------------------------------------------------------------------------

  @doc """
  Backward pass completo.

  Retorna %{dblocks, dw_head} — gradientes de todos os parâmetros.
  """
  def backward(cache, model) do
    {d_outputs, dw_head} =
      language_head_backward(cache.final_output, cache.lh_cache, model.w_head)

    {_, dblocks} =
      Enum.zip(model.blocks_params, cache.block_caches)
      |> Enum.reverse()
      |> Enum.reduce({d_outputs, []}, fn {params, block_cache}, {dy, dparams_acc} ->
        {dx, dparams} = transformer_block_backward(block_cache, params, dy)
        {dx, [dparams | dparams_acc]}
      end)

    %{dblocks: dblocks, dw_head: dw_head}
  end

  # ---------------------------------------------------------------------------
  # Helpers
  # ---------------------------------------------------------------------------

  defp unzip5(list) do
    {a, b, c, d, e} =
      Enum.reduce(list, {[], [], [], [], []}, fn {a, b, c, d, e}, {as, bs, cs, ds, es} ->
        {[a | as], [b | bs], [c | cs], [d | ds], [e | es]}
      end)

    {Enum.reverse(a), Enum.reverse(b), Enum.reverse(c), Enum.reverse(d), Enum.reverse(e)}
  end
end
