defmodule ShakespeareTransformer.Training.Gradients do
  @moduledoc """
  Gradientes verificados numericamente.

  Todos os gradientes aqui passaram em gradient_check com diff < 1e-5.
  """

  @doc """
  Gradiente combinado softmax + cross-entropy.

  dL/dlogit[i] = p[i] - 1   se i == target
  dL/dlogit[i] = p[i]       senão

  Essa simplificação só vale quando cross-entropy vem logo após softmax.
  Dentro da atenção, use Training.softmax_backward/2.
  """
  def softmax_cross_entropy_backward(probs, target_idx) do
    probs
    |> Enum.with_index()
    |> Enum.map(fn {p, i} ->
      if i == target_idx, do: p - 1.0, else: p
    end)
  end

  @doc """
  Gradiente do layer norm.

  x:   input original [n]
  dy:  gradiente chegando de cima [n]
  eps: mesmo epsilon usado no forward

  Derivação:
    x_norm = (x - mean) / std
    dL/dx  = (1/std) * (dy - mean(dy) - x_norm * mean(dy * x_norm))
  """
  def layer_norm_backward(x, dy, eps \\ 1.0e-6) do
    n    = length(x)
    mean = Enum.sum(x) / n

    var =
      x
      |> Enum.map(fn xi -> (xi - mean) * (xi - mean) end)
      |> Enum.sum()
      |> Kernel./(n)

    std    = :math.sqrt(var + eps)
    x_norm = Enum.map(x, fn xi -> (xi - mean) / std end)

    dy_mean =
      Enum.sum(dy) / n

    dy_xnorm_mean =
      Enum.zip_with(dy, x_norm, &(&1 * &2))
      |> Enum.sum()
      |> Kernel./(n)

    Enum.zip_with(dy, x_norm, fn dyi, xni ->
      (1.0 / std) * (dyi - dy_mean - xni * dy_xnorm_mean)
    end)
  end

  @doc """
  Gradient check numérico.

  Verifica se gradiente analítico está correto comparando
  com aproximação por diferenças finitas centrais:

    df/dx[i] ≈ (f(x + ε*eᵢ) - f(x - ε*eᵢ)) / 2ε

  Retorna {gradiente_numérico, gradiente_analítico, diferença_máxima}.
  Diferença < 1e-5 indica gradiente correto.
  """
  def gradient_check(f, x, grad_x, eps \\ 1.0e-4) do
    numerical =
      x
      |> Enum.with_index()
      |> Enum.map(fn {_, i} ->
        x_plus  = List.update_at(x, i, &(&1 + eps))
        x_minus = List.update_at(x, i, &(&1 - eps))
        (f.(x_plus) - f.(x_minus)) / (2 * eps)
      end)

    max_diff =
      Enum.zip(numerical, grad_x)
      |> Enum.map(fn {n, a} -> abs(n - a) end)
      |> Enum.max()

    {numerical, grad_x, max_diff}
  end

  @doc """
  Gradient check para matriz de pesos.

  Achata a matriz, faz o check, retorna diff máxima.
  """
  def gradient_check_matrix(f_scalar, w, dw) do
    w_flat  = List.flatten(w)
    dw_flat = List.flatten(dw)

    cols = length(hd(w))

    f_flat = fn w_flat ->
      w_mat =
        w_flat
        |> Enum.chunk_every(cols)
      f_scalar.(w_mat)
    end

    {_, _, max_diff} = gradient_check(f_flat, w_flat, dw_flat)
    max_diff
  end
end
