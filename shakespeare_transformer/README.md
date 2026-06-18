# ShakespeareTransformer

## Inspired on

[Inspirado em](../studyplan/0002.PURE_ELIXIR.md)

## Running

```sh
iex -S mix
```

## Executing

```
alias ShakespeareTransformer.Tokenizer
alias ShakespeareTransformer.Training.{Trainer, Forward}

# initialize model
text = File.read!("priv/input.txt")
{_chars, c2i, i2c} = Tokenizer.build_vocab(text)

model = Trainer.init_model(map_size(c2i), 32, 2, 16, 2)

# or load model
model = File.read!("priv/model.bin") |> :erlang.binary_to_term()

tokens  = Tokenizer.encode("To be or not", c2i)
targets = Enum.drop(tokens, 1)
inputs  = Enum.drop(tokens, -1)

{loss, _cache} = Forward.forward(inputs, targets, model)
IO.puts("loss: #{loss}")

model = Trainer.train(model, text, c2i,
  epochs:    50,
  lr:        1.0e-2,
  seq_len:   16,
  log_every: 10
)

# Learning
{micros, model} = :timer.tc(fn ->
  Trainer.train(model, text, c2i,
    epochs:    50,
    lr:        1.0e-2,
    seq_len:   128,
    log_every: 10
  )
end)

# 3 hours
{micros, model} = :timer.tc(fn ->
  Trainer.train(model, text, c2i,
    epochs:    3_000,
    lr:        1.0e-3,
    seq_len:   128,
    log_every: 50
  )
end)

seconds = micros / 1_000_000
IO.puts("Tempo total: #{Float.round(seconds, 2)}s (#{Float.round(seconds / 60, 2)} min)")

# save model
:erlang.term_to_binary(model) |> then(&File.write!("priv/model.bin", &1))

{micros, output} = :timer.tc(fn ->
  Trainer.generate(model, "To be", c2i, i2c, 200)
end)

seconds = micros / 1_000_000
IO.puts("Tempo total: #{Float.round(seconds, 2)}s (#{Float.round(seconds / 60, 2)} min)")
```


{micros, output} = :timer.tc(fn ->
  Trainer.generate(model, "To be", c2i, i2c, 20)
end)

## Results

2 Trainings

Parameters on each training:

{micros, model} = :timer.tc(fn ->
  Trainer.train(model, text, c2i,
    epochs:    3_000,
    lr:        1.0e-3,
    seq_len:   128,
    log_every: 50
  )
end)

Average time of training:

iex(71)> IO.puts("Tempo total: #{Float.round(seconds, 2)}s (#{Float.round(seconds / 60, 2)} min)")
Tempo total: 9589.42s (159.82 min)

Last 5 loss

epoch 2800/3000 — loss: 3.1846
epoch 2850/3000 — loss: 3.0967
epoch 2900/3000 — loss: 3.1685
epoch 2950/3000 — loss: 3.1731
epoch 3000/3000 — loss: 3.3241

Inference time and results:

20 tokens

iex(73)> {micros, output} = :timer.tc(fn ->
...(73)>   Trainer.generate(model, "To be", c2i, i2c, 20)
...(73)> end)
{430548, "To be uertsllnlta\nh,sdnh "}

iex(76)> IO.puts("Tempo total: #{Float.round(seconds, 2)}s (#{Float.round(seconds / 60, 2)} min)")
Tempo total: 0.43s (0.01 min)

200 tokens

iex(78)> {micros, output} = :timer.tc(fn ->
...(78)>   Trainer.generate(model, "To be", c2i, i2c, 200)
...(78)> end)
{132475750,
 "To beemrt r\ns e; \n\nwt : totbgdiatst MoedoTtbivsrR:uwhtso,yolret s\nmuum  HooeEnob,kmis d ta oknl c\nu;uh\naO,Iet\neem Xsah t tVhIihnl \n h. ti,,'nw mhni nsiAaaioa?tOaIi .B.ltteet  l tX'd:e   stot stmunhdddnpoeo"}

seconds = micros / 1_000_000
IO.puts("Tempo total: #{Float.round(seconds, 2)}s (#{Float.round(seconds / 60, 2)} min)")
