defmodule ShakespeareTransformer.ModelRegistry do
  @moduledoc """
  Registro central de modelos por personagem, em ETS.

  Convenção de arquivos, dado um corpus em "priv/kobold/grik.txt":

    priv/kobold/grik.txt                     — input (corpus)
    priv/kobold/grik_bpe_tokenizer.json       — tokenizer treinado (auto)
    priv/kobold/nx_model_grik.axon            — pesos do treino (auto)
    priv/kobold/grik.struct                   — struct completa (auto, pra reload rápido)

  O id do personagem é sempre a string derivada do nome do arquivo
  (sem extensão) — "grik.txt" → id "grik".

  Não chame este módulo diretamente — use ShakespeareTransformer
  como entrypoint (new_model/1, load_model/1, train_model/2,
  model_generate/3, autoload_dir/1).
  """

  use GenServer

  alias ShakespeareTransformer.{CharacterModel, BpeTokenizer, NxModel, NxTrainer}

  @table :character_models

  # ---------------------------------------------------------------------------
  # GenServer — entra na árvore de supervisão como child de verdade.
  # ---------------------------------------------------------------------------

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @impl true
  def init(opts) do
    case :ets.whereis(@table) do
      :undefined -> :ets.new(@table, [:named_table, :public, :set, read_concurrency: true])
      _tid -> :ok
    end

    autoload_dir = Keyword.get(opts, :autoload_dir, System.get_env("SHAKESPEARE_AUTOLOAD_DIR"))

    if autoload_dir do
      send(self(), {:autoload, autoload_dir})
    end

    {:ok, %{}}
  end

  @impl true
  def handle_info({:autoload, dir}, state) do
    autoload_dir(dir)
    {:noreply, state}
  end

  # ---------------------------------------------------------------------------
  # Convenção de paths
  # ---------------------------------------------------------------------------

  @doc "Deriva o id (string) a partir do path do corpus. 'priv/kobold/grik.txt' -> 'grik'"
  def id_from_path(corpus_path) do
    corpus_path
    |> Path.basename()
    |> Path.rootname()
  end

  @doc "Gera os paths convencionados a partir do diretório e id."
  def conventional_paths(dir, id) do
    %{
      tokenizer: Path.join(dir, "#{id}_tokenizer.json"),
      weights:   Path.join(dir, "nx_model_#{id}.axon"),
      struct:    Path.join(dir, "#{id}.struct")
    }
  end

  # ---------------------------------------------------------------------------
  # new_model — cria um modelo do zero a partir de um corpus
  # ---------------------------------------------------------------------------

  @doc """
  Cria um novo modelo do zero a partir de um arquivo de corpus.
  O id é derivado automaticamente do nome do arquivo, e os paths
  de tokenizer/pesos/struct seguem a convenção do diretório do corpus.

  opts (todas opcionais):
    mode       — :bpe (default) ou :word. Veja BpeTokenizer moduledoc
                 pra entender o trade-off entre os dois.
    vocab_size — default 400. Só relevante pra mode: :bpe.
    d_model    — default 48
    n_heads    — default 2
    n_blocks   — default 2
    seq_len    — default 48

  Retorna o %CharacterModel{} já registrado na ETS (pesos ainda
  aleatórios — chame train_model/2 em seguida).
  """
  def new_model(corpus_path, opts \\ []) do
    mode       = Keyword.get(opts, :mode, :word)
    vocab_size = Keyword.get(opts, :vocab_size, 400)
    d_model    = Keyword.get(opts, :d_model, 48)
    n_heads    = Keyword.get(opts, :n_heads, 2)
    n_blocks   = Keyword.get(opts, :n_blocks, 2)
    seq_len    = Keyword.get(opts, :seq_len, 48)

    id  = id_from_path(corpus_path)
    dir = Path.dirname(corpus_path)
    paths = conventional_paths(dir, id)

    {:ok, tokenizer} = BpeTokenizer.train(corpus_path, mode: mode, vocab_size: vocab_size)
    BpeTokenizer.save(tokenizer, paths.tokenizer)
    real_vocab_size = BpeTokenizer.vocab_size(tokenizer)

    model =
      NxModel.build(
        vocab_size: real_vocab_size,
        d_model:    d_model,
        n_heads:    n_heads,
        n_blocks:   n_blocks,
        seq_len:    seq_len
      )

    character = %CharacterModel{
      id: id,
      model: model,
      params: Axon.ModelState.empty(),
      tokenizer: tokenizer,
      idx_to_char: nil,
      hyperparams: %{
        vocab_size: real_vocab_size,
        d_model:    d_model,
        n_heads:    n_heads,
        n_blocks:   n_blocks,
        seq_len:    seq_len,
        tokenizer_mode: mode
      },
      corpus_path: corpus_path,
      tokenizer_path: paths.tokenizer,
      weights_path: paths.weights,
      struct_path: paths.struct,
      metadata: %{
        total_epochs: 0,
        last_loss: nil,
        created_at: DateTime.utc_now()
      }
    }

    put(character)
    save_to_disk(character)
    character
  end

  # ---------------------------------------------------------------------------
  # load_model — carrega um modelo já treinado do disco
  # ---------------------------------------------------------------------------

  @doc """
  Carrega um %CharacterModel{} previamente salvo (.struct) e registra
  na ETS. model, tokenizer e params são todos reconstruídos a partir
  de dados portáveis — veja save_to_disk/1 pra entender por quê.
  """
  def load_model(struct_path) do
    character =
      struct_path
      |> File.read!()
      |> :erlang.binary_to_term()

    model =
      NxModel.build(
        vocab_size: character.hyperparams.vocab_size,
        d_model:    character.hyperparams.d_model,
        n_heads:    character.hyperparams.n_heads,
        n_blocks:   character.hyperparams.n_blocks,
        seq_len:    character.hyperparams.seq_len
      )

    {:ok, tokenizer} = BpeTokenizer.load(character.tokenizer_path)

    params =
      if File.exists?(character.weights_path) do
        character.weights_path |> File.read!() |> Nx.deserialize()
      else
        Axon.ModelState.empty()
      end

    character = %CharacterModel{character | model: model, tokenizer: tokenizer, params: params}

    put(character)

    IO.puts("loaded character #{inspect(character.id)} from #{struct_path}")

    character
  end

  @doc """
  Varre um diretório (ou padrão glob) procurando arquivos `*.struct`
  e carrega todos em paralelo, registrando cada um na ETS.

      autoload_dir("priv/kobold")        # varre só priv/kobold/*.struct
      autoload_dir("priv/*")             # varre priv/<qualquer_pasta>/*.struct
      autoload_dir("priv/**")            # varre recursivamente
  """
  def autoload_dir(pattern) do
    struct_files =
      pattern
      |> Path.join("*.struct")
      |> Path.wildcard()

    IO.puts("autoload: #{length(struct_files)} struct(s) encontrada(s) em #{pattern}")

    loaded =
      struct_files
      |> Enum.map(fn path -> Task.async(fn -> load_model(path) end) end)
      |> Task.await_many(:infinity)

    IO.puts("autoload: #{length(loaded)} personagem(ns) carregado(s)")

    loaded
  end

  # ---------------------------------------------------------------------------
  # train_model — treina (ou continua treinando) um modelo já registrado
  # ---------------------------------------------------------------------------

  @doc """
  Treina o modelo de um personagem já registrado na ETS (por id).
  Atualiza params e metadata, persiste tudo em disco (pesos + struct),
  e atualiza a entrada na ETS.

  opts: as mesmas de NxTrainer.train/4 (epochs, lr, batch_size, log_every, save_every)
  """
  def train_model(id, opts \\ []) do
    character = fetch!(id)
    text = File.read!(character.corpus_path)

    tmp_save_path = character.weights_path <> ".tmp"

    train_opts =
      opts
      |> Keyword.put_new(:seq_len, character.hyperparams.seq_len)
      |> Keyword.put(:save_path, tmp_save_path)
      |> Keyword.put(:initial_params, character.params)

    {model, new_params} = NxTrainer.train(character.model, text, character.tokenizer, train_opts)

    epochs_run = Keyword.get(opts, :epochs, 1000)

    updated = %CharacterModel{
      character
      | model: model,
        params: new_params,
        metadata: %{
          character.metadata
          | total_epochs: character.metadata.total_epochs + epochs_run
        }
    }

    put(updated)
    save_to_disk(updated)
    File.rm(tmp_save_path)
    updated
  end

  # ---------------------------------------------------------------------------
  # model_generate — gera texto a partir do modelo de um personagem
  # ---------------------------------------------------------------------------

  @doc """
  Gera texto usando o modelo de um personagem registrado (por id).

  opts: as mesmas de NxTrainer.generate/7 (temperature, top_k), mais:
    clean — se true (default), usa generate_clean (corta frase incompleta final)
  """
  def model_generate(id, prompt, n_tokens, opts \\ []) do
    character = fetch!(id)
    seq_len   = character.hyperparams.seq_len

    clean = Keyword.get(opts, :clean, true)

    gen_opts =
      opts
      |> Keyword.drop([:clean])
      |> Keyword.put(:seq_len, seq_len)

    if clean do
      NxTrainer.generate_clean(
        character.model, character.params, prompt,
        character.tokenizer, character.idx_to_char, n_tokens,
        gen_opts
      )
    else
      NxTrainer.generate(
        character.model, character.params, prompt,
        character.tokenizer, character.idx_to_char, n_tokens,
        gen_opts
      )
    end
  end

  # ---------------------------------------------------------------------------
  # Acesso direto à ETS
  # ---------------------------------------------------------------------------

  def fetch!(id) do
    id = to_string(id)

    case :ets.lookup(@table, id) do
      [{^id, character}] -> character
      [] -> raise "Nenhum modelo registrado para #{inspect(id)}. Use new_model/2, load_model/1 ou autoload_dir/1."
    end
  end

  def fetch(id) do
    id = to_string(id)

    case :ets.lookup(@table, id) do
      [{^id, character}] -> {:ok, character}
      [] -> :not_found
    end
  end

  def put(%CharacterModel{id: id} = character) do
    :ets.insert(@table, {id, character})
    character
  end

  def list_ids do
    :ets.tab2list(@table) |> Enum.map(fn {id, _char} -> id end)
  end

  def delete(id) do
    :ets.delete(@table, to_string(id))
  end

  # ---------------------------------------------------------------------------
  # Persistência
  # ---------------------------------------------------------------------------

  @doc """
  Salva a struct (hiperparâmetros, metadata, paths) no struct_path, e
  os pesos (params) separadamente via Nx.serialize no weights_path.

  Nem tokenizer, nem params, nem model entram no binário genérico do
  struct_path: todos os três carregam referências a recursos nativos
  que não sobrevivem a um restart do BEAM se serializados com
  :erlang.term_to_binary puro. Reconstruídos em load_model/1.
  """
  def save_to_disk(%CharacterModel{} = character) do
    File.mkdir_p!(Path.dirname(character.struct_path))

    weights_binary = Nx.serialize(character.params)
    File.write!(character.weights_path, weights_binary)

    struct_binary =
      character
      |> Map.put(:tokenizer, nil)
      |> Map.put(:params, nil)
      |> Map.put(:model, nil)
      |> :erlang.term_to_binary()

    File.write!(character.struct_path, struct_binary)
    character
  end

  def save_to_disk(id) do
    id |> fetch!() |> save_to_disk()
  end
end
