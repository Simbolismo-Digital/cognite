defmodule ShakespeareTransformer.NxTrainer do
  @moduledoc """
  Loop de treino e geração usando Nx + Axon.

  Mesma arquitetura, mesmos hiperparâmetros que a versão Elixir puro —
  só o substrato muda. Onde antes você tinha listas e loops manuais,
  agora tem tensores e Nx.Defn compilando pra CPU via EXLA.

  Uso:

    alias ShakespeareTransformer.{Tokenizer, NxModel, NxTrainer}

    text = File.read!("priv/input.txt")
    {_chars, c2i, i2c} = Tokenizer.build_vocab(text)
    vocab_size = map_size(c2i)

    model = NxModel.build(
      vocab_size: vocab_size,
      d_model:    32,
      n_heads:    2,
      n_blocks:   2,
      seq_len:    128
    )

    {trained_model, trained_params} =
      NxTrainer.train(model, text, c2i,
        vocab_size: vocab_size,
        seq_len:    128,
        epochs:     3000,
        lr:         1.0e-3,
        batch_size: 32
      )

    NxTrainer.generate(trained_model, trained_params, "To be", c2i, i2c, 200)
  """

  alias ShakespeareTransformer.Tokenizer

  # ---------------------------------------------------------------------------
  # Preparação dos dados em batches
  # ---------------------------------------------------------------------------

  @doc """
  Gera um batch de sequências aleatórias do texto.

  Retorna {inputs, targets} como tensores Nx.
    inputs:  [batch_size, seq_len]
    targets: [batch_size, seq_len]
  """
  def random_batch(tokens_tensor, total, seq_len, batch_size) do
    max_start = total - seq_len - 1
    starts = for _ <- 1..batch_size, do: :rand.uniform(max_start) - 1

    # constrói índices [batch_size, seq_len+1] e usa gather —
    # evita Nx.slice com offset dinâmico, que força recompilação por valor
    idx_matrix =
      starts
      |> Enum.map(fn start -> Enum.to_list(start..(start + seq_len)) end)
      |> Nx.tensor(type: :s64)

    chunks = Nx.take(tokens_tensor, idx_matrix)

    inputs  = Nx.slice_along_axis(chunks, 0, seq_len, axis: 1)
    targets = Nx.slice_along_axis(chunks, 1, seq_len, axis: 1)

    {inputs, targets}
  end

  # ---------------------------------------------------------------------------
  # Loop de treino
  # ---------------------------------------------------------------------------

  @doc """
  Treina o modelo.

  opts:
    vocab_size, seq_len, epochs, lr, batch_size, log_every, save_every, save_path,
    initial_params — passe um Axon.ModelState (de NxTrainer.load_params/1)
                     pra continuar treino de onde parou. Se omitido, começa do zero.
  """
  def train(model, text, char_to_idx, opts \\ []) do
    seq_len        = Keyword.fetch!(opts, :seq_len)
    epochs         = Keyword.get(opts, :epochs,     1000)
    lr             = Keyword.get(opts, :lr,         1.0e-3)
    batch_size     = Keyword.get(opts, :batch_size, 32)
    log_every      = Keyword.get(opts, :log_every,  50)
    save_every     = Keyword.get(opts, :save_every, log_every)
    save_path      = Keyword.get(opts, :save_path,  "priv/nx_model.axon")
    initial_params = Keyword.get(opts, :initial_params, load_or_empty(save_path))

    all_tokens = Tokenizer.encode(text, char_to_idx)
    tokens_tensor = Nx.tensor(all_tokens, type: :s64)
    total = length(all_tokens)

    IO.puts("Iniciando treino Nx — #{total} tokens, seq_len=#{seq_len}, batch_size=#{batch_size}, lr=#{lr}")

    loss_fn = fn y_true, y_pred ->
      vocab = Nx.axis_size(y_pred, -1)
      class_indices = Nx.iota({1, 1, vocab})
      y_true_oh = Nx.equal(Nx.new_axis(y_true, -1), class_indices)
      Axon.Losses.categorical_cross_entropy(y_true_oh, y_pred,
        reduction: :mean,
        from_logits: true
      )
    end

    optimizer =
      Polaris.Updates.clip_by_global_norm(max_norm: 1.0)
      |> Polaris.Updates.compose(Polaris.Optimizers.adam(learning_rate: lr))

    loop =
      model
      |> Axon.Loop.trainer(loss_fn, optimizer, log: 0)
      |> Axon.Loop.log(
        fn state -> "epoch #{state.epoch}, batch #{state.iteration}, loss #{:io_lib.format("~.4f", [Nx.to_number(state.step_state.loss)])}\n" end,
        event: :iteration_completed,
        filter: [every: log_every]
      )
      |> Axon.Loop.handle_event(:iteration_completed, fn state ->
        iteration = state.iteration + 1

        if save_every && rem(iteration, save_every) == 0 do
          binary = Nx.serialize(state.step_state[:model_state] || state.step_state)
          File.write!(save_path, binary)
        end

        {:continue, state}
      end)

    # gera um stream de batches aleatórios
    data_stream =
      Stream.repeatedly(fn ->
        random_batch(tokens_tensor, total, seq_len, batch_size)
      end)

    final_state =
      loop
      |> Axon.Loop.run(data_stream, initial_params, epochs: 1, iterations: epochs, compiler: EXLA)

    {model, final_state}
  end

  def load_or_empty(path) do
    if File.exists?(path) do
      load_params(path)
    else
      Axon.ModelState.empty()
    end
  end

  # ---------------------------------------------------------------------------
  # Geração de texto
  # ---------------------------------------------------------------------------

  @doc """
  Gera texto autorregressivo a partir de um prompt.
  """
  def generate(model, params, prompt, char_to_idx, idx_to_char, n_tokens, opts \\ []) do
    seq_len     = Keyword.fetch!(opts, :seq_len)
    temperature = Keyword.get(opts, :temperature, 1.0)

    {_init_fn, predict_fn} = Axon.build(model, compiler: EXLA)

    initial_tokens = Tokenizer.encode(prompt, char_to_idx)

    final_tokens =
      Enum.reduce(1..n_tokens, initial_tokens, fn _, tokens ->
        # left-padding: mantém a última posição sempre fixa em seq_len - 1,
        # evitando recompilação por offset dinâmico no slice
        context = left_pad_or_trim(tokens, seq_len)

        input_tensor =
          context
          |> Nx.tensor(type: :s64)
          |> Nx.new_axis(0)

        logits = predict_fn.(params, %{"tokens" => input_tensor})

        # última posição é sempre seq_len - 1, fixo — sem recompilação
        last_logits =
          logits
          |> Nx.slice([0, seq_len - 1, 0], [1, 1, Nx.axis_size(logits, -1)])
          |> Nx.squeeze()
          |> Nx.divide(temperature)

        probs = Axon.Activations.softmax(last_logits)
        next_token = sample(probs)

        tokens ++ [next_token]
      end)

    generated = Enum.drop(final_tokens, length(initial_tokens))
    prompt <> Tokenizer.decode(generated, idx_to_char)
  end

  defp left_pad_or_trim(tokens, seq_len) do
    cond do
      length(tokens) >= seq_len ->
        Enum.take(tokens, -seq_len)

      true ->
        padding = List.duplicate(0, seq_len - length(tokens))
        padding ++ tokens
    end
  end

  defp sample(probs) do
    probs_list = Nx.to_flat_list(probs)
    r = :rand.uniform()

    {idx, _} =
      probs_list
      |> Enum.with_index()
      |> Enum.reduce_while({0, 0.0}, fn {p, i}, {_, cum} ->
        new_cum = cum + p
        if new_cum >= r, do: {:halt, {i, new_cum}}, else: {:cont, {i, new_cum}}
      end)

    idx
  end

  # ---------------------------------------------------------------------------
  # Persistência
  # ---------------------------------------------------------------------------

  def save_params(params, path) do
    binary = Nx.serialize(params)
    File.write!(path, binary)
  end

  def load_params(path) do
    path
    |> File.read!()
    |> Nx.deserialize()
  end
end
