defmodule ShakespeareTransformer.BpeTokenizer do
  @moduledoc """
  Tokenizador por subpalavra (BPE) usando a lib `Tokenizers`.

  Diferente do tokenizador char-level original — aqui o vocabulário
  é aprendido a partir do corpus, agrupando sequências de caracteres
  frequentes em tokens únicos. "the" vira 1 token em vez de 3.

  Uso:

    {:ok, tokenizer} = BpeTokenizer.train("priv/input.txt", vocab_size: 4000)
    BpeTokenizer.save(tokenizer, "priv/bpe_tokenizer.json")

    # depois, pra carregar sem re-treinar:
    {:ok, tokenizer} = BpeTokenizer.load("priv/bpe_tokenizer.json")

    {:ok, ids} = BpeTokenizer.encode(tokenizer, "To be or not to be")
    text = BpeTokenizer.decode(tokenizer, ids)
  """

  alias Tokenizers.{Tokenizer, Model, Trainer}

  @doc """
  Treina um tokenizador BPE no texto do arquivo dado.

  opts:
    vocab_size: tamanho do vocabulário a aprender (default 4000)
    min_frequency: frequência mínima pra um par virar token (default 2)
  """
  def train(text_path, opts \\ []) do
    vocab_size    = Keyword.get(opts, :vocab_size, 4000)
    min_frequency = Keyword.get(opts, :min_frequency, 2)

    {:ok, model} = Model.BPE.empty()
    {:ok, tokenizer} = Tokenizer.init(model)

    tokenizer = Tokenizer.set_pre_tokenizer(tokenizer, Tokenizers.PreTokenizer.byte_level())
    tokenizer = Tokenizer.set_decoder(tokenizer, Tokenizers.Decoder.byte_level())

    {:ok, trainer} =
      Trainer.bpe(
        vocab_size: vocab_size,
        min_frequency: min_frequency,
        special_tokens: ["<pad>", "<unk>", "<bos>", "<eos>"]
      )

    {:ok, tokenizer} = Tokenizer.train_from_files(tokenizer, [text_path], trainer: trainer)

    {:ok, tokenizer}
  end

  @doc "Salva o tokenizador treinado em disco (formato JSON)"
  def save(tokenizer, path) do
    Tokenizer.save(tokenizer, path)
  end

  @doc "Carrega um tokenizador previamente treinado e salvo"
  def load(path) do
    Tokenizer.from_file(path)
  end

  @doc "Codifica texto em lista de IDs de tokens"
  def encode(tokenizer, text) do
    case Tokenizer.encode(tokenizer, text) do
      {:ok, encoding} -> {:ok, Tokenizers.Encoding.get_ids(encoding)}
      error -> error
    end
  end

  @doc "Decodifica lista de IDs de volta pra texto"
  def decode(tokenizer, ids) do
    {:ok, text} = Tokenizer.decode(tokenizer, ids)
    text
  end

  @doc "Tamanho do vocabulário aprendido"
  def vocab_size(tokenizer) do
    Tokenizer.get_vocab_size(tokenizer)
  end
end
