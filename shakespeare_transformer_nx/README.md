# ShakespeareTransformer

## Inspired on

[Inspirado em](../studyplan/0002.PURE_ELIXIR.md)

## Running

```sh
iex -S mix
```

## Executing

```
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

{micros, {model, params}} = :timer.tc(fn ->
  NxTrainer.train(model, text, c2i,
    seq_len:    128,
    epochs:     3000,
    lr:         1.0e-3,
    batch_size: 32,
    log_every:  50
  )
end)

IO.puts("Tempo: #{micros / 1_000_000 / 60} min")

NxTrainer.generate(model, params, "To be", c2i, i2c, 200, seq_len: 128)
```

## Results

1 Training

Parameters on each training:

{micros, {model, params}} = :timer.tc(fn ->
  NxTrainer.train(model, text, c2i,
    seq_len:    128,
    epochs:     3000,
    lr:         1.0e-3,
    batch_size: 32,
    log_every:  50
  )
end)

Last five loss:
epoch 0, batch 2750, loss 2.3173
epoch 0, batch 2800, loss 2.3141
epoch 0, batch 2850, loss 2.3110
epoch 0, batch 2900, loss 2.3079
epoch 0, batch 2950, loss 2.3048


Average time of training:
{168396165,
 {#Axon
    inputs: %{"tokens" => {nil, 128}}
    outputs: "language_head"
    nodes: 33
  >,
  #Axon.ModelState
    Parameters: 29441 (117.76 KB)
    Trainable Parameters: 29441 (117.76 KB)
    Trainable State: 0, (0 B)
  >}}

iex(12)> IO.puts("Tempo: #{micros / 1_000_000 / 60} min")
Tempo: 3.0509893 min
:ok

Result of inference was almost instant:

iex(11)> NxTrainer.generate(model, params, "To be", c2i, i2c, 200, seq_len: 128)
"To be wnaut Hath gooubhtWhin seade r nos mort'\nis if 'e talespeory ave hime garris.\nI SThe blost;\nLef feit fio, bealms eat fores now, my my so, so of or metto dave\nI marl ase loves tard will uthe tad sip d"

Comparisson with handmade:
Far faster. Real words appearing.

Second training:

epoch 0, batch 2700, loss 2.0545
epoch 0, batch 2750, loss 2.0534
epoch 0, batch 2800, loss 2.0524
epoch 0, batch 2850, loss 2.0514
epoch 0, batch 2900, loss 2.0505
epoch 0, batch 2950, loss 2.0496
{177299783,
 {#Axon<
    inputs: %{"tokens" => {nil, 128}}
    outputs: "language_head"
    nodes: 37
  >,
  #Axon.ModelState<
    Parameters: 29449 (117.80 KB)
    Trainable Parameters: 29441 (117.76 KB)
    Trainable State: 8, (32 B)
  >}}
iex(13)> IO.puts("Tempo: #{micros / 1_000_000 / 60} min")
Tempo: 2.954996383333333 min
:ok
iex(14)> 
nil
iex(15)> NxTrainer.generate(model, params, "To be", c2i, i2c, 200, seq_len: 128)
"To best no ENCE:\nForort ale arvest yere.\n\nKANTIOUS:\nNo, your nenot toortould, the own deit,\nAnd he king Duard site do\nder at hat nother affors. Gozentem lold,\nAnd fail I dor, fave foule sirok do frecics:\nS"


## Results 2

alias ShakespeareTransformer.{Tokenizer, NxModel, NxTrainer}

text = File.read!("priv/input.txt")
{_chars, c2i, i2c} = Tokenizer.build_vocab(text)
vocab_size = map_size(c2i)

model = NxModel.build(
  vocab_size: vocab_size,
  d_model:    64,
  n_heads:    4,
  n_blocks:   3,
  seq_len:    192
)

{micros, {model, params}} = :timer.tc(fn ->
  NxTrainer.train(model, text, c2i,
    seq_len:    192,
    epochs:     3000,
    lr:         3.0e-4,
    batch_size: 48,
    log_every:  200,
    save_path:  "priv/nx_model_2.axon"
  )
end)

IO.puts("Tempo: #{micros / 1_000_000 / 60} min")

NxTrainer.generate(model, params, "To be", c2i, i2c, 200, seq_len: 192)

