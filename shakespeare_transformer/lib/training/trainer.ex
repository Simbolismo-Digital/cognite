defmodule ShakespeareTransformer.Training.Trainer do
  @moduledoc """
  Loop de treino completo.

  Uso:

    model = Trainer.init_model(vocab_size, d_model, n_heads, d_head, n_blocks)
    model = Trainer.train(model, text, char_to_idx, epochs: 100, lr: 1.0e-3, seq_len: 64)
    Trainer.generate(model, "To be", char_to_idx, idx_to_char, 200)
  """

  alias ShakespeareTransformer.{Embedding, TransformerBlock, LanguageHead, Training}
  alias ShakespeareTransformer.Training.{Backward,  Forward}

  # ---------------------------------------------------------------------------
  # Inicialização do modelo
  # ---------------------------------------------------------------------------

  @doc """
  Inicializa modelo com pesos aleatórios.

  Retorna %{table, blocks_params, w_head, d_model, vocab_size}
  """
  def init_model(vocab_size, d_model, n_heads, d_head, n_blocks) do
    %{
      vocab_size:    vocab_size,
      d_model:       d_model,
      n_heads:       n_heads,
      d_head:        d_head,
      n_blocks:      n_blocks,
      table:         Embedding.init(vocab_size, d_model),
      blocks_params: for(_ <- 1..n_blocks, do: TransformerBlock.init(d_model, n_heads, d_head)),
      w_head:        LanguageHead.init(d_model, vocab_size)
    }
  end

  # ---------------------------------------------------------------------------
  # Loop de treino
  # ---------------------------------------------------------------------------

  @doc """
  Treina o modelo por N épocas.

  opts:
    epochs:  número de épocas (default 10)
    lr:      learning rate (default 1e-3)
    seq_len: tamanho da sequência (default 64)
    log_every: imprime loss a cada N épocas (default 1)
  """
  def train(model, text, char_to_idx, opts \\ []) do
    epochs    = Keyword.get(opts, :epochs,    10)
    lr        = Keyword.get(opts, :lr,        1.0e-3)
    seq_len   = Keyword.get(opts, :seq_len,   64)
    log_every = Keyword.get(opts, :log_every, 1)

    all_tokens = ShakespeareTransformer.Tokenizer.encode(text, char_to_idx)
    total      = length(all_tokens)

    IO.puts("Iniciando treino — #{total} tokens, seq_len=#{seq_len}, lr=#{lr}")

    Enum.reduce(1..epochs, model, fn epoch, model ->
      # amostra posição aleatória
      start   = :rand.uniform(total - seq_len - 1)
      chunk   = Enum.slice(all_tokens, start, seq_len + 1)
      tokens  = Enum.drop(chunk, -1)
      targets = Enum.drop(chunk, 1)

      # forward
      {loss, cache} = Forward.forward(tokens, targets, model)

      # backward
      grads = Backward.backward(cache, model)

      # atualiza pesos
      model = update_model(model, grads, lr)

      if rem(epoch, log_every) == 0 do
        :erlang.term_to_binary(model) |> then(&File.write!("priv/model.bin", &1))
        IO.puts("epoch #{epoch}/#{epochs} — loss: #{Float.round(loss, 4)}")
      end

      model
    end)
  end

  # ---------------------------------------------------------------------------
  # Geração de texto
  # ---------------------------------------------------------------------------

  @doc """
  Gera texto autorregressivo a partir de um prompt.

  prompt:      string inicial
  n_tokens:    quantos tokens gerar
  temperature: 1.0 = normal, < 1.0 = mais determinístico, > 1.0 = mais aleatório
  """
  def generate(model, prompt, char_to_idx, idx_to_char, n_tokens, temperature \\ 1.0) do
    initial_tokens = ShakespeareTransformer.Tokenizer.encode(prompt, char_to_idx)

    {_, final_tokens} =
      Enum.reduce(1..n_tokens, {initial_tokens, initial_tokens}, fn _, {context, all_tokens} ->
        # forward só pra pegar logits — sem calcular loss
        vetores = Embedding.encode_sequence(context, model.table, model.d_model)

        output =
          Enum.reduce(model.blocks_params, vetores, fn params, input ->
            {out, _cache} = Forward.transformer_block(input, params)
            out
          end)

        # pega último token
        last_vec  = List.last(output)
        logits    = LanguageHead.project(last_vec, model.w_head)

        # aplica temperature
        scaled_logits = Training.scale_vec(logits, 1.0 / temperature)
        probs         = LanguageHead.softmax(scaled_logits)

        # amostra próximo token
        next_token = sample(probs)

        # mantém contexto com tamanho máximo pra não explodir memória
        max_ctx    = 128
        new_ctx    = if length(context) >= max_ctx do
          Enum.drop(context, 1) ++ [next_token]
        else
          context ++ [next_token]
        end

        {new_ctx, all_tokens ++ [next_token]}
      end)

    # decodifica
    generated = Enum.drop(final_tokens, length(initial_tokens))
    prompt <> ShakespeareTransformer.Tokenizer.decode(generated, idx_to_char)
  end

  # ---------------------------------------------------------------------------
  # Atualização de pesos
  # ---------------------------------------------------------------------------

  defp update_model(model, grads, lr) do
    updated_blocks =
      Enum.zip(model.blocks_params, grads.dblocks)
      |> Enum.map(fn {params, dparams} ->
        Training.update_block_params(params, dparams, lr)
      end)

    updated_w_head =
      Training.update_weights(model.w_head, grads.dw_head, lr)

    %{model | blocks_params: updated_blocks, w_head: updated_w_head}
  end

  # ---------------------------------------------------------------------------
  # Sampling
  # ---------------------------------------------------------------------------

  defp sample(probs) do
    r = :rand.uniform()
    {idx, _} =
      probs
      |> Enum.with_index()
      |> Enum.reduce_while({0, 0.0}, fn {p, i}, {_, cum} ->
        new_cum = cum + p
        if new_cum >= r, do: {:halt, {i, new_cum}}, else: {:cont, {i, new_cum}}
      end)
    idx
  end
end
