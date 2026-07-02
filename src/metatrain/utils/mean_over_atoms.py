from typing import List

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

from . import torch_jit_script_unless_coverage


@torch_jit_script_unless_coverage
def mean_over_atoms(tensor_map: TensorMap) -> TensorMap:
    """
    Like ``sum_over_atoms``, but AVERAGES per-atom contributions over the atoms
    of each system. This is the correct (intensive) reduction for per-structure
    properties that do not scale with system size, e.g. a d-band centre or a
    band gap. Using this instead of ``sum_over_atoms`` removes the implicit
    1/N inductive bias that the summing convention forces on the network.

    :param tensor_map: The TensorMap to average over.
    :return: A new TensorMap with one sample per system.
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

        summed = torch.zeros(
            [n_systems] + list(block.values.shape[1:]), device=device, dtype=dtype
        )
        summed.index_add_(0, system_samples, block.values)

        # Per-system atom counts (as float, for division).
        counts = torch.zeros(n_systems, device=device, dtype=dtype)
        counts.index_add_(
            0, system_samples, torch.ones_like(system_samples, dtype=dtype)
        )
        counts = counts.clamp(min=1.0)  # guard against empty systems

        # Broadcast counts over the trailing (component/property) dims.
        counts_view = counts
        for _ in range(summed.dim() - 1):
            counts_view = counts_view.unsqueeze(-1)
        averaged = summed / counts_view

        new_blocks.append(
            TensorBlock(
                values=averaged,
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