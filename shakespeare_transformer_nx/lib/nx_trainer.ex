defmodule ShakespeareTransformer.NxTrainer do
  @moduledoc """
  Loop de treino e geração usando Nx + Axon.

  A garantia de "só palavras do corpus" vem do tokenizer agora
  (mode: :word em BpeTokenizer), não de filtro pós-geração — se o
  tokenizer é word-level, o modelo fisicamente não tem como escolher
  um token que não existia no vocabulário de treino.

  Uso:

    alias ShakespeareTransformer.{BpeTokenizer, NxModel, NxTrainer}

    text = File.read!("priv/input.txt")
    {:ok, tokenizer} = BpeTokenizer.train("priv/input.txt", mode: :word)
    vocab_size = BpeTokenizer.vocab_size(tokenizer)

    model = NxModel.build(
      vocab_size: vocab_size, d_model: 48, n_heads: 2, n_blocks: 2, seq_len: 48
    )

    {model, params} =
      NxTrainer.train(model, text, tokenizer,
        seq_len: 48, epochs: 3000, lr: 3.0e-4, batch_size: 16
      )

    NxTrainer.generate(model, params, "Welcome", tokenizer, nil, 30,
      seq_len: 48, temperature: 0.7, top_k: 10
    )
  """

  alias ShakespeareTransformer.{Tokenizer, BpeTokenizer}

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

  vocabulary: pode ser
    - um char_to_idx map (tokenizador char-level original), ou
    - um %Tokenizers.Tokenizer{} (BPE ou word-level) — detectado automaticamente

  opts:
    seq_len, epochs, lr, batch_size, log_every, save_every, save_path,
    initial_params — passe um Axon.ModelState (de NxTrainer.load_params/1)
                     pra continuar treino de onde parou. Se omitido, começa do zero.
  """
  def train(model, text, vocabulary, opts \\ []) do
    seq_len        = Keyword.fetch!(opts, :seq_len)
    epochs         = Keyword.get(opts, :epochs,     1000)
    lr             = Keyword.get(opts, :lr,         1.0e-3)
    batch_size     = Keyword.get(opts, :batch_size, 32)
    log_every      = Keyword.get(opts, :log_every,  50)
    save_every     = Keyword.get(opts, :save_every, log_every)
    save_path      = Keyword.get(opts, :save_path,  "priv/nx_model.axon")
    initial_params = Keyword.get(opts, :initial_params, load_or_empty(save_path))

    all_tokens = encode_text(text, vocabulary)
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
        fn state -> "epoch #{state.epoch}, batch #{state.iteration}, loss #{:io_lib.format("~.4f", [Nx.to_number(state.step_state.loss)])}, now=#{DateTime.utc_now() |> DateTime.to_iso8601()}\n" end,
        event: :iteration_completed,
        filter: [every: log_every]
      )
      |> Axon.Loop.handle_event(:iteration_completed, fn state ->
        iteration = state.iteration

        if save_every && rem(iteration, save_every) == 0 do
          binary = Nx.serialize(state.step_state[:model_state] || state.step_state)
          File.write!(save_path, binary)
        end

        {:continue, state}
      end)

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

  vocabulary: char_to_idx map (char-level) ou %Tokenizers.Tokenizer{}
  (BPE ou word-level) — mesmo objeto passado no train/4. Pra char-level,
  também precisa de idx_to_char.

  opts:
    seq_len      — obrigatório, mesmo usado no treino
    temperature  — default 1.0. < 1.0 = mais conservador, > 1.0 = mais criativo
    top_k        — default nil. Quando definido, restringe a amostragem
                   aos k tokens mais prováveis a cada passo.
  """
  def generate(model, params, prompt, vocabulary, idx_to_char_or_nil, n_tokens, opts \\ []) do
    seq_len     = Keyword.fetch!(opts, :seq_len)
    temperature = Keyword.get(opts, :temperature, 1.0)
    top_k       = Keyword.get(opts, :top_k, nil)

    {_init_fn, predict_fn} = Axon.build(model, compiler: EXLA)

    initial_tokens = encode_text(prompt, vocabulary)

    final_tokens =
      Enum.reduce(1..n_tokens, initial_tokens, fn _, tokens ->
        context = left_pad_or_trim(tokens, seq_len)

        input_tensor =
          context
          |> Nx.tensor(type: :s64)
          |> Nx.new_axis(0)

        logits = predict_fn.(params, %{"tokens" => input_tensor})

        last_logits =
          logits
          |> Nx.slice([0, seq_len - 1, 0], [1, 1, Nx.axis_size(logits, -1)])
          |> Nx.squeeze()
          |> Nx.divide(temperature)

        probs = Axon.Activations.softmax(last_logits)
        next_token = sample(probs, top_k)

        tokens ++ [next_token]
      end)

    generated = Enum.drop(final_tokens, length(initial_tokens))
    prompt <> decode_tokens(generated, vocabulary, idx_to_char_or_nil)
  end

  @doc """
  Corta o texto no último ponto final/exclamação/interrogação completo,
  descartando qualquer fragmento de frase incompleta no final.
  """
  def truncate_to_last_sentence(text) do
    case Regex.scan(~r/.*?[.!?]/s, text) do
      [] -> text
      matches ->
        matches
        |> Enum.map(fn [m] -> m end)
        |> Enum.join("")
        |> String.trim()
    end
  end

  @doc """
  Gera texto e descarta qualquer fragmento de frase incompleta no final.

  Por padrão retorna TODAS as sentenças completas geradas. Passe
  :sentences pra limitar a um número específico.
  """
  def generate_clean(model, params, prompt, vocabulary, idx_to_char_or_nil, n_tokens, opts \\ []) do
    sentences = Keyword.get(opts, :sentences, nil)
    gen_opts  = Keyword.drop(opts, [:sentences])

    full_text = generate(model, params, prompt, vocabulary, idx_to_char_or_nil, n_tokens, gen_opts)

    cleaned = truncate_to_last_sentence(full_text)

    case sentences do
      nil ->
        String.trim(cleaned)

      n ->
        cleaned
        |> String.split(~r/(?<=[.!?])\s+/)
        |> Enum.take(n)
        |> Enum.join(" ")
        |> String.trim()
    end
  end

  # ---------------------------------------------------------------------------
  # Despacho de tokenização — char-level (map) vs Tokenizers (BPE ou word-level)
  # ---------------------------------------------------------------------------

  defp encode_text(text, %Tokenizers.Tokenizer{} = tokenizer) do
    {:ok, ids} = BpeTokenizer.encode(tokenizer, text)
    ids
  end

  defp encode_text(text, char_to_idx) when is_map(char_to_idx) do
    Tokenizer.encode(text, char_to_idx)
  end

  defp decode_tokens(ids, %Tokenizers.Tokenizer{} = tokenizer, _idx_to_char) do
    BpeTokenizer.decode(tokenizer, ids)
  end

  defp decode_tokens(ids, char_to_idx, idx_to_char) when is_map(char_to_idx) do
    Tokenizer.decode(ids, idx_to_char)
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

  # ---------------------------------------------------------------------------
  # Sampling — com suporte opcional a top_k
  # ---------------------------------------------------------------------------

  @doc """
  Amostra um token a partir da distribuição de probabilidades.

  top_k: quando nil (default), amostra sobre toda a distribuição.
         quando inteiro, restringe a amostragem aos k tokens mais
         prováveis, renormalizando as probabilidades só entre eles.
  """
  def sample(probs, top_k \\ nil) do
    probs_list = Nx.to_flat_list(probs)
    indexed    = Enum.with_index(probs_list)

    candidates =
      if top_k do
        indexed
        |> Enum.sort_by(fn {p, _i} -> p end, :desc)
        |> Enum.take(top_k)
      else
        indexed
      end

    total = candidates |> Enum.map(fn {p, _i} -> p end) |> Enum.sum()
    r = :rand.uniform() * total

    {idx, _} =
      candidates
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
