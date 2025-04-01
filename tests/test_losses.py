import torch
from topology_lib.losses import topological_preservation_loss

def test_topological_preservation_loss():
    latent_data = torch.rand((10, 2), requires_grad=True)
    loss = topological_preservation_loss(latent_data, homology_dim=1, weight=1.0)
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.requires_grad, "Loss should require gradients"

def test_no_homology_dim():
    latent_data = torch.rand((10, 2), requires_grad=True)
    loss = topological_preservation_loss(latent_data, homology_dim=5, weight=1.0)
    assert loss.item() == 0.0, "Loss should be zero for non-existent homology dimension"