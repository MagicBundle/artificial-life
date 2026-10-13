"""pytest plugin: run MLX on the CPU device so tests never touch the GPU (evaluation arms own it)."""
import mlx.core as mx


def pytest_configure(config):
    mx.set_default_device(mx.cpu)
