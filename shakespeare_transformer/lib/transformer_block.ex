defmodule ShakespeareTransformer.TransformerBlock do
  @moduledoc """
  5
  alias ShakespeareTransformer.TransformerBlock

  params = TransformerBlock.init(d_model, n_heads, d_head)
  block_output = TransformerBlock.forward(vetores, params)

  IO.puts("entrada: \#{length(vetores)} vetores de \#{length(hd(vetores))} dims")
  IO.puts("saída:   \#{length(block_output)} vetores de \#{length(hd(block_output))} dims")

  params2 = TransformerBlock.init(d_model, n_heads, d_head)
  block_output2 = TransformerBlock.forward(block_output, params2)

  IO.puts("após 2 blocos: \#{length(block_output2)} vetores de \#{length(hd(block_output2))} dims")
  """
  alias ShakespeareTransformer.{Attention, FeedForward}

  @doc "Layer normalization de um vetor"
  def layer_norm(vec, eps \\ 1.0e-6) do
    n    = length(vec)
    mean = Enum.sum(vec) / n
    var  = vec
           |> Enum.map(fn x -> (x - mean) * (x - mean) end)
           |> Enum.sum()
           |> Kernel./(n)

    Enum.map(vec, fn x ->
      (x - mean) / :math.sqrt(var + eps)
    end)
  end

  @doc "Soma dois vetores — residual connection"
  def add(v1, v2) do
    Enum.zip_with(v1, v2, &(&1 + &2))
  end

  @doc """
  Forward pass de um bloco completo.

  inputs: lista de vetores [d_model]
  params: %{heads, w_out, w1, b1, w2, b2}
  """
  def forward(inputs, %{heads: heads, w_out: w_out,
                         w1: w1, b1: b1, w2: w2, b2: b2}) do
    # --- sublayer 1: multi-head attention ---
    normed1  = Enum.map(inputs, &layer_norm/1)
    attended = Attention.multi_head_attention(normed1, heads, w_out)
    # residual
    x1 = Enum.zip_with(inputs, attended, &add/2)

    # --- sublayer 2: feed-forward ---
    normed2 = Enum.map(x1, &layer_norm/1)
    ff_out  = FeedForward.forward_sequence(normed2, w1, b1, w2, b2)
    # residual
    x2 = Enum.zip_with(x1, ff_out, &add/2)

    x2
  end

  @doc "Inicializa todos os parâmetros de um bloco"
  def init(d_model, n_heads, d_head) do
    %{
      heads: Attention.init_heads(n_heads, d_model, d_head),
      w_out: Attention.init_weights(n_heads * d_head, d_model),
      w1:    nil,
      b1:    nil,
      w2:    nil,
      b2:    nil
    }
    |> Map.merge(
      FeedForward.init(d_model)
      |> then(fn {w1, b1, w2, b2} ->
        %{w1: w1, b1: b1, w2: w2, b2: b2}
      end)
    )
  end
end
