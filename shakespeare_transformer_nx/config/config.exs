import Config

config :nx, default_backend: EXLA.Backend

config :exla, :clients,
  host: [platform: :host]

config :exla, :preferred_clients, [:host]
