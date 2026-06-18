defmodule ShakespeareTransformer.FeedForward do
  @moduledoc """
  4
  alias ShakespeareTransformer.FeedForward

  {w1, b1, w2, b3} = FeedForward.init(d_model)

  ff_output = FeedForward.forward_sequence(output, w1, b1, w2, b3)

  IO.puts("entrada: \#{length(output)} vetores de \#{length(hd(output))} dims")
  IO.puts("saída:   \#{length(ff_output)} vetores de \#{length(hd(ff_output))} dims")
  """

  @doc "ReLU — zera negativos"
  def relu(vec) do
    Enum.map(vec, &max(0.0, &1))
  end

  @doc "Multiplicação vetor × matriz + bias"
  def linear(vec, weights, bias) do
    weights
    |> transpose()
    |> Enum.zip(bias)
    |> Enum.map(fn {col, b} ->
      dot_product(vec, col) + b
    end)
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

  @doc """
  Forward pass completo.
  x: vetor [d_model]
  retorna: vetor [d_model]
  """
  def forward(x, w1, b1, w2, b2) do
    x
    |> linear(w1, b1)   # [d_model] → [4*d_model]
    |> relu()            # zera negativos
    |> linear(w2, b2)   # [4*d_model] → [d_model]
  end

  @doc "Aplica feed-forward em cada token da sequência"
  def forward_sequence(inputs, w1, b1, w2, b2) do
    Enum.map(inputs, &forward(&1, w1, b1, w2, b2))
  end

  @doc "Inicializa pesos e bias"
  def init(d_model) do
    d_ff = 4 * d_model
    scale = :math.sqrt(2.0 / d_model)

    w1 = for _ <- 1..d_model do
      for _ <- 1..d_ff do
        (:rand.uniform() - 0.5) * scale
      end
    end

    b1 = for _ <- 1..d_ff, do: 0.0

    w2 = for _ <- 1..d_ff do
      for _ <- 1..d_model do
        (:rand.uniform() - 0.5) * scale
      end
    end

    b2 = for _ <- 1..d_model, do: 0.0

    {w1, b1, w2, b2}
  end
end
