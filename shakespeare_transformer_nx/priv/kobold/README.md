# BPE Tokenizer

```
alias ShakespeareTransformer.BpeTokenizer

{:ok, tokenizer} = BpeTokenizer.train("priv/kobold/first.txt", vocab_size: 600)
BpeTokenizer.save(tokenizer, "priv/kobold/first_bpe_tokenizer.json")
```

Train model
```
alias ShakespeareTransformer.{BpeTokenizer, NxModel, NxTrainer}

text = File.read!("priv/kobold/first.txt")
{:ok, tokenizer} = BpeTokenizer.load("priv/kobold/first_bpe_tokenizer.json")   # carrega em vez de treinar
vocab_size = BpeTokenizer.vocab_size(tokenizer)

model = NxModel.build(
  vocab_size: vocab_size,
  d_model:    32,
  n_heads:    2,
  n_blocks:   2,
  seq_len:    32
)

{micros, {model, params}} = :timer.tc(fn ->
  NxTrainer.train(model, text, tokenizer,
    seq_len:    32,
    epochs:     3000,
    lr:         3.0e-4,
    batch_size: 16,
    log_every:  200,
    save_path:  "priv/kobold/nx_model_kobold.axon"
  )
end)

IO.puts("Tempo: #{micros / 1_000_000 / 60} min")

NxTrainer.generate(model, params, "Grik hungry", tokenizer, nil, 30, seq_len: 32, temperature: 0.7, top_k: 10)
```