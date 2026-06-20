defmodule ShakespeareTransformer.CharacterModel do
  @moduledoc """
  Struct que encapsula um modelo completo de um personagem —
  arquitetura, pesos, tokenizer e hiperparâmetros, tudo junto.

  Isso é o que vive na ETS: uma unidade autocontida que pode ser
  carregada, treinada e usada pra gerar texto sem precisar reconstruir
  nada externamente.
  """

  @enforce_keys [:id, :model, :params, :tokenizer, :hyperparams]
  defstruct [
    :id,           # string — derivada do nome do arquivo de corpus (ex: "grik")
    :model,        # %Axon{} — grafo da arquitetura
    :params,       # %Axon.ModelState{} — pesos treinados
    :tokenizer,    # %Tokenizers.Tokenizer{} ou char_to_idx map
    :idx_to_char,  # só usado se tokenizer for char-level. nil pra BPE
    :hyperparams,  # %{vocab_size, d_model, n_heads, n_blocks, seq_len}
    :corpus_path,  # path do arquivo de corpus usado pra treinar (pra retreinar depois)
    :tokenizer_path, # priv/kobold/grik_bpe_tokenizer.json
    :weights_path,   # priv/kobold/nx_model_grik.axon
    :struct_path,    # priv/kobold/grik.struct — onde a struct inteira é salva
    :metadata      # mapa livre — total de épocas treinadas, loss final, criado_em, etc
  ]

  @type t :: %__MODULE__{
          id: String.t(),
          model: Axon.t(),
          params: Axon.ModelState.t(),
          tokenizer: Tokenizers.Tokenizer.t() | map(),
          idx_to_char: map() | nil,
          hyperparams: map(),
          corpus_path: String.t() | nil,
          tokenizer_path: String.t(),
          weights_path: String.t(),
          struct_path: String.t(),
          metadata: map()
        }
end
