defmodule ShakespeareTransformer.Application do
  @moduledoc """
  Callback de Application — sobe o ModelRegistry como child supervisionado
  automaticamente quando o projeto inicia, e já varre "priv/**" em busca
  de personagens salvos (*.struct), carregando todos em paralelo.

  Pra apontar pra outro diretório, defina a env var SHAKESPEARE_AUTOLOAD_DIR
  antes de subir o projeto (ex: SHAKESPEARE_AUTOLOAD_DIR="priv/kobold").
  """

  use Application

  @impl true
  def start(_type, _args) do
    children = [
      {ShakespeareTransformer.ModelRegistry,
       autoload_dir: System.get_env("SHAKESPEARE_AUTOLOAD_DIR", "priv/**")}
    ]

    opts = [strategy: :one_for_one, name: ShakespeareTransformer.Supervisor]
    Supervisor.start_link(children, opts)
  end
end
