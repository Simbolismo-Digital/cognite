defmodule ShakespeareTransformer.Training do
  @moduledoc """
  Backpropagation manual e gradient descent.
  """

  # ---------------------------------------------------------------------------
  # Primitivas matemáticas
  # ---------------------------------------------------------------------------

  def dot_product(v1, v2) do
    Enum.zip_with(v1, v2, &(&1 * &2)) |> Enum.sum()
  end

  def transpose(matrix) do
    matrix |> Enum.zip() |> Enum.map(&Tuple.to_list/1)
  end

  def mat_mul(a, b) do
    b_t = transpose(b)
    Enum.map(a, fn row_a ->
      Enum.map(b_t, fn col_b -> dot_product(row_a, col_b) end)
    end)
  end

  def scale_vec(vec, s), do: Enum.map(vec, &(&1 * s))

  def add_vec(v1, v2), do: Enum.zip_with(v1, v2, &(&1 + &2))

  def add_mat(m1, m2) do
    Enum.zip_with(m1, m2, fn r1, r2 ->
      Enum.zip_with(r1, r2, &(&1 + &2))
    end)
  end

  def zeros(n), do: List.duplicate(0.0, n)

  def zeros_mat(rows, cols) do
    for _ <- 1..rows, do: List.duplicate(0.0, cols)
  end

  def outer_product(x, dy) do
    Enum.map(x, fn xi -> Enum.map(dy, fn dyi -> xi * dyi end) end)
  end

  # ---------------------------------------------------------------------------
  # Forward helpers
  # ---------------------------------------------------------------------------

  def linear_forward(x, w, b) do
    w
    |> transpose()
    |> Enum.zip(b)
    |> Enum.map(fn {col, bi} -> dot_product(x, col) + bi end)
  end

  # ---------------------------------------------------------------------------
  # Gradientes básicos
  # ---------------------------------------------------------------------------

  def linear_backward_weights(x, dy), do: outer_product(x, dy)

  def linear_backward_input(dy, w) do
    Enum.map(w, fn row -> dot_product(row, dy) end)
  end

  def relu_backward(dy, x) do
    Enum.zip_with(dy, x, fn dyi, xi -> if xi > 0, do: dyi, else: 0.0 end)
  end

  def bias_backward(dy), do: dy

  def softmax_backward(p, dy) do
    dot = dot_product(dy, p)
    Enum.zip_with(p, dy, fn pi, dyi -> pi * (dyi - dot) end)
  end

  # ---------------------------------------------------------------------------
  # Feed-forward backward
  # ---------------------------------------------------------------------------

  def feed_forward_backward(x, h, a, dy, w1, _b1, w2, _b2) do
    da  = linear_backward_input(dy, w2)
    dw2 = linear_backward_weights(a, dy)
    db2 = dy

    dh  = relu_backward(da, h)
    dx  = linear_backward_input(dh, w1)
    dw1 = linear_backward_weights(x, dh)
    db1 = dh

    {dx, dw1, db1, dw2, db2}
  end

  # ---------------------------------------------------------------------------
  # Gradient descent
  # ---------------------------------------------------------------------------

  def update_weights(w, dw, lr) do
    Enum.zip_with(w, dw, fn row, drow ->
      Enum.zip_with(row, drow, fn wi, dwi -> wi - lr * dwi end)
    end)
  end

  def update_bias(b, db, lr) do
    Enum.zip_with(b, db, fn bi, dbi -> bi - lr * dbi end)
  end

  def update_block_params(params, dparams, lr) do
    heads =
      Enum.zip(params.heads, dparams.heads)
      |> Enum.map(fn {{wq, wk, wv}, {dwq, dwk, dwv}} ->
        {
          update_weights(wq, dwq, lr),
          update_weights(wk, dwk, lr),
          update_weights(wv, dwv, lr)
        }
      end)

    %{params |
      heads: heads,
      w_out: update_weights(params.w_out, dparams.w_out, lr),
      w1:    update_weights(params.w1,    dparams.w1,    lr),
      b1:    update_bias(   params.b1,    dparams.b1,    lr),
      w2:    update_weights(params.w2,    dparams.w2,    lr),
      b2:    update_bias(   params.b2,    dparams.b2,    lr)
    }
  end

  # ---------------------------------------------------------------------------
  # Helpers legados (mantidos pra compatibilidade)
  # ---------------------------------------------------------------------------

  def add_grads(g1, g2), do: add_mat(g1, g2)
  def add_vec_legacy(v1, v2), do: add_vec(v1, v2)
end
