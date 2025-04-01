import torch
from topology_lib.autoencoder import SimpleAutoencoder

def test_autoencoder_forward():
    model = SimpleAutoencoder(input_dim=2, hidden_dim=16, code_dim=2)
    x = torch.rand((5, 2))
    out, z = model(x)
    assert out.shape == (5, 2), "Output shape should match input shape"
    assert z.shape == (5, 2), "Encoded shape should match code_dim"