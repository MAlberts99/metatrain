from typing import List

import torch
from e3nn import o3
from metatensor.torch import Labels, TensorBlock, TensorMap

from . import torch_jit_script_unless_coverage


class AtomAttentionPooling(torch.nn.Module):
    """
    Segment (per-system) attention pooling over atoms, for INTENSIVE per-structure
    targets (d-band centre, band gap).

    A rotation-invariant score is produced per atom from the equivariant node
    features, softmaxed over the atoms of each structure, and used as convex
    weights. Because the weights sum to 1 within each structure, the pooled value
    is size-invariant (intensive), and if the network drives the scores to be
    uniform this reduces exactly to mean pooling — so it's a strict generalisation
    of ``mean_over_atoms``.

    Only the 0e (scalar) part of ``irreps_in`` feeds the score (via ``o3.Linear``
    to a scalar irrep), so the attention weights are invariant by construction.
    """

    def __init__(self, irreps_in: o3.Irreps, hidden: int = 64):
        super().__init__()
        # Mixed irreps -> hidden scalars; e3nn.Linear only routes the 0e parts to
        # a 0e output, guaranteeing invariance of the resulting scores.
        self.proj = o3.Linear(irreps_in, o3.Irreps([(hidden, (0, 1))]))
        self.act = torch.nn.SiLU()
        self.score = torch.nn.Linear(hidden, 1)
        torch.nn.init.zeros_(self.score.weight)
        torch.nn.init.zeros_(self.score.bias)


    def weights(
        self,
        node_feats: torch.Tensor,
        system_index: torch.Tensor,
        n_systems: int,
    ) -> torch.Tensor:
        """
        :param node_feats: [n_atoms, irreps_in.dim] equivariant node features.
        :param system_index: [n_atoms] the system each atom belongs to.
        :param n_systems: number of systems in the batch.
        :return: [n_atoms, 1] convex attention weights (per-system softmax).
        """
        h = self.act(self.proj(node_feats))          # [n_atoms, hidden] (invariant)
        logits = self.score(h)                        # [n_atoms, 1]

        # Numerically stable per-system softmax over atoms.
        neg_inf = torch.full(
            (n_systems, 1), -1.0e30, device=logits.device, dtype=logits.dtype
        )
        seg_max = neg_inf.scatter_reduce(
            0, system_index.unsqueeze(-1), logits, reduce="amax", include_self=True
        )
        shifted = logits - seg_max[system_index]
        exp = torch.exp(shifted)
        seg_sum = torch.zeros(
            (n_systems, 1), device=logits.device, dtype=logits.dtype
        )
        seg_sum.index_add_(0, system_index, exp)
        return exp / (seg_sum[system_index] + 1.0e-12)


@torch_jit_script_unless_coverage
def weighted_sum_over_atoms(
    tensor_map: TensorMap, weights: torch.Tensor
) -> TensorMap:
    """
    Like ``sum_over_atoms``, but weights each atom's contribution by ``weights``
    before summing. With convex (softmax) weights this is attention pooling: an
    intensive, learned-weighted average over the atoms of each system.

    Assumes scalar targets (no ``components``); ``weights`` is [n_atoms, 1] and
    broadcasts over the property dimension.

    :param tensor_map: per-atom TensorMap to pool.
    :param weights: [n_atoms, 1] per-atom weights, aligned to the samples order.
    """
    new_blocks: List[TensorBlock] = []
    for block in tensor_map.blocks():
        device = block.values.device
        dtype = block.values.dtype
        system_samples = block.samples.column("system")
        if system_samples.numel() == 0:
            n_systems = 0
        else:
            n_systems = int(system_samples.max()) + 1

        weighted = block.values * weights.to(dtype)   # [n_atoms, n_props]
        out = torch.zeros(
            [n_systems] + list(block.values.shape[1:]), device=device, dtype=dtype
        )
        out.index_add_(0, system_samples, weighted)

        new_blocks.append(
            TensorBlock(
                values=out,
                samples=Labels(
                    names=["system"],
                    values=torch.arange(
                        n_systems, device=device, dtype=torch.int32
                    ).reshape(-1, 1),
                    assume_unique=True,
                ),
                components=block.components,
                properties=block.properties,
            )
        )
    return TensorMap(keys=tensor_map.keys, blocks=new_blocks)