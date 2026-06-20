defmodule ShakespeareTransformer do
  @moduledoc """
  Entrypoint público do projeto.

  Cada personagem é um modelo transformer pequeno e independente,
  identificado pelo nome do arquivo de corpus que o originou.
  Convenção de arquivos, dado "priv/kobold/grik.txt":

    priv/kobold/grik.txt                  — input, escrito por você
    priv/kobold/grik_bpe_tokenizer.json   — tokenizer treinado (gerado)
    priv/kobold/nx_model_grik.axon        — pesos do treino (gerado)
    priv/kobold/grik.struct               — struct completa (gerada, pra reload rápido)

  Todos os modelos registrados vivem numa tabela ETS em memória,
  acessados por id (string derivada do nome do corpus).

  O boot da aplicação já varre "priv/**" automaticamente e carrega
  todo personagem salvo (*.struct) em paralelo — não precisa chamar
  nada manualmente pra isso. Veja ShakespeareTransformer.Application.

  ## Uso

      # primeira vez — cria e treina um personagem novo
      ShakespeareTransformer.new_model("priv/kobold/grik.txt",
        vocab_size: 600, d_model: 48, n_heads: 2, n_blocks: 2, seq_len: 48
      )

      ShakespeareTransformer.train_model("grik",
        epochs: 3000, lr: 3.0e-4, batch_size: 16, log_every: 200
      )

      ShakespeareTransformer.model_generate("grik", "Grik hungry", 30,
        temperature: 0.7, top_k: 10, best: 5
      )

      # próxima vez subindo o projeto — já vem carregado sozinho
      ShakespeareTransformer.list_characters()
      #=> ["grik", "elara", "merchant"]
  """

  alias ShakespeareTransformer.ModelRegistry

  @doc """
  O registro de personagens já sobe automaticamente como child
  supervisionado no boot da aplicação — não precisa chamar isso
  manualmente em uso normal. Útil só em testes que sobem o
  ModelRegistry isoladamente, fora da árvore de supervisão padrão.
  """
  defdelegate start(opts \\ []), to: ModelRegistry, as: :start_link

  @doc """
  Cria um novo personagem do zero a partir de um arquivo de corpus.
  Id derivado automaticamente do nome do arquivo.
  """
  defdelegate new_model(corpus_path, opts \\ []), to: ModelRegistry

  @doc "Carrega um personagem já salvo (.struct) pelo path."
  defdelegate load_model(struct_path), to: ModelRegistry

  @doc """
  Varre um diretório por arquivos `*.struct` e carrega todos em
  paralelo. Forma recomendada de subir o elenco inteiro no boot.
  """
  defdelegate autoload_dir(dir), to: ModelRegistry

  @doc "Treina (ou continua treinando) o personagem identificado por id."
  defdelegate train_model(id, opts \\ []), to: ModelRegistry

  @doc "Gera texto a partir do personagem identificado por id."
  defdelegate model_generate(id, prompt, n_tokens, opts \\ []), to: ModelRegistry

  @doc "Lista os ids de todos os personagens atualmente carregados."
  defdelegate list_characters(), to: ModelRegistry, as: :list_ids

  @doc "Busca a struct completa de um personagem pelo id."
  defdelegate get_character(id), to: ModelRegistry, as: :fetch!

  @doc "Persiste manualmente o estado atual de um personagem em disco."
  defdelegate save_character(id), to: ModelRegistry, as: :save_to_disk
end
