import Config

System.put_env("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=true --xla_force_host_platform_device_count=16")
