import torch
import numpy as np
from ripser import ripser

def topological_preservation_loss(latent_data: torch.Tensor,
                                  homology_dim: int = 1,
                                  weight: float = 1.0) -> torch.Tensor:
    """
    Считает «топологическую потерю» (на самом деле – добавку к функции потерь),
    которая ПООЩРЯЕТ сохранение 1D-дыр (т.е. больших циклов) в latent_data.

    Если мы хотим *максимизировать* сумму длины жизни 1D-классов,
    то в функцию потерь вносится *отрицательное* значение этих длины жизни,
    чтобы градиент пытался увеличить (death - birth).

    Параметры:
      - latent_data:  (N, d) тензор, где N - число точек, d - размерность латентного пространства.
      - homology_dim:  размерность гомологии (как правило 1, если хотим 1D-дыры).
      - weight:  вес добавки. Итоговый вклад = -1 * weight * sum_of_lifetimes.
    """
    # Переводим в NumPy (ripser работает с NumPy-массивами)
    data_np = latent_data.detach().cpu().numpy()

    # Вычисляем диаграммы устойчивой гомологии
    dgms = ripser(data_np, maxdim=homology_dim)['dgms']

    # Извлекаем диаграмму для интересующей размерности
    if homology_dim < len(dgms):
        dgm_dim = dgms[homology_dim]
    else:
        # Нет такой размерности
        return torch.tensor(0.0, device=latent_data.device, requires_grad=True)

    if dgm_dim.shape[0] == 0:
        # Если нет баров (классов) в этой размерности, возвращаем 0
        return torch.tensor(0.0, device=latent_data.device, requires_grad=True)

    # Суммарное время существования всех классов: sum (death - birth)
    lifetimes = dgm_dim[:, 1] - dgm_dim[:, 0]
    sum_lifetime = np.sum(lifetimes)

    # Так как мы ХОТИМ увеличить sum_lifetime, в функцию потерь добавим "- sum_lifetime"
    # (чтобы градиент был направлен на увеличение этой величины).
    # Умножаем на weight.
    loss_val = - weight * torch.tensor(sum_lifetime,
                                       dtype=torch.float32,
                                       device=latent_data.device)
    return loss_val
