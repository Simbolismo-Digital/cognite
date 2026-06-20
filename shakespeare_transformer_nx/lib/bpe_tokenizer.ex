defmodule ShakespeareTransformer.BpeTokenizer do
  @moduledoc """
  Tokenizador configurável usando a lib `Tokenizers`. Suporta dois modos:

    :bpe   — subpalavra (default). "the" pode virar 1 token; palavras
             novas quebram em fragmentos conhecidos. Mais flexível,
             mas pode gerar combinações que nunca existiram no corpus
             (ex: "ants" a partir de fragmentos de outras palavras).

    :word  — palavra inteira. Vocabulário é EXATAMENTE o conjunto de
             palavras do corpus de treino. O modelo fisicamente não
             consegue gerar uma palavra fora desse conjunto — a garantia
             vem da arquitetura, não de filtro/retry depois da geração.
             Trade-off: sem flexão livre de sufixo/prefixo; palavra que
             não apareceu no corpus vira <unk> sempre.

  Uso:

    {:ok, tok} = BpeTokenizer.train("priv/input.txt", mode: :bpe, vocab_size: 600)
    {:ok, tok} = BpeTokenizer.train("priv/input.txt", mode: :word)

    BpeTokenizer.save(tok, "priv/tokenizer.json")
    {:ok, tok} = BpeTokenizer.load("priv/tokenizer.json")

    {:ok, ids} = BpeTokenizer.encode(tok, "To be or not to be")
    text = BpeTokenizer.decode(tok, ids)
  """

  alias Tokenizers.{Tokenizer, Model, Trainer}

  @special_tokens ["<pad>", "<unk>", "<bos>", "<eos>"]

  @doc """
  Treina um tokenizador no texto do arquivo dado.

  opts:
    mode: :bpe (default) ou :word
    vocab_size: tamanho do vocabulário a aprender — só relevante pra :bpe,
                ignorado em :word (vocabulário é o que existir no corpus)
    min_frequency: frequência mínima pra um token entrar no vocabulário
                   (default 1 pra :word, 2 pra :bpe)
  """
  def train(text_path, opts \\ []) do
    mode = Keyword.get(opts, :mode, :bpe)

    case mode do
      :bpe  -> train_bpe(text_path, opts)
      :word -> train_word(text_path, opts)
    end
  end

  defp train_bpe(text_path, opts) do
    vocab_size    = Keyword.get(opts, :vocab_size, 600)
    min_frequency = Keyword.get(opts, :min_frequency, 2)

    {:ok, model} = Model.BPE.empty()
    {:ok, tokenizer} = Tokenizer.init(model)

    tokenizer = Tokenizer.set_pre_tokenizer(tokenizer, Tokenizers.PreTokenizer.byte_level())
    tokenizer = Tokenizer.set_decoder(tokenizer, Tokenizers.Decoder.byte_level())

    {:ok, trainer} =
      Trainer.bpe(
        vocab_size: vocab_size,
        min_frequency: min_frequency,
        special_tokens: @special_tokens
      )

    Tokenizer.train_from_files(tokenizer, [text_path], trainer: trainer)
  end

  defp train_word(text_path, opts) do
    min_frequency = Keyword.get(opts, :min_frequency, 1)

    {:ok, model} = Model.WordLevel.empty()
    {:ok, tokenizer} = Tokenizer.init(model)

    # pre-tokenizer por whitespace: cada palavra separada por espaço vira
    # uma unidade candidata a token. Sem isso o WordLevel não sabe onde
    # cortar o texto bruto antes de mapear pro vocabulário.
    tokenizer = Tokenizer.set_pre_tokenizer(tokenizer, Tokenizers.PreTokenizer.whitespace())

    {:ok, trainer} =
      Trainer.wordlevel(
        min_frequency: min_frequency,
        special_tokens: @special_tokens
      )

    Tokenizer.train_from_files(tokenizer, [text_path], trainer: trainer)
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

  @doc """
  Decodifica lista de IDs de volta pra texto.

  No modo :word, o pre-tokenizer de whitespace separa pontuação como
  token próprio (ex: "today" "."), o que o decode junta de volta com
  espaço (ex: "today ."). Aqui removemos esse espaço espúrio antes de
  pontuação comum, deixando o texto com formatação natural.
  """
  def decode(tokenizer, ids) do
    {:ok, text} = Tokenizer.decode(tokenizer, ids)
    fix_punctuation_spacing(text)
  end

  defp fix_punctuation_spacing(text) do
    text
    |> String.replace(~r/\s+([.,!?;:'")])/u, "\\1")
    |> String.replace(~r/(['"(])\s+/u, "\\1")
  end

  @doc "Tamanho do vocabulário aprendido"
  def vocab_size(tokenizer) do
    Tokenizer.get_vocab_size(tokenizer)
  end
end
