# topology_lib/__init__.py

from .losses import topological_preservation_loss
from .autoencoder import SimpleAutoencoder

__all__ = [
    "topological_preservation_loss",
    "SimpleAutoencoder"
]

# Версия пакета
__version__ = "0.1.0"