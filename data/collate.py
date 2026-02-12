"""Custom collation for variable-length batches.

Handles padding of agent/map polylines and metadata aggregation.
"""

import torch


def mtr_collate_fn(batch):
    """Collate function for MTR-Lite dataset.

    Filters None samples, stacks fixed-size tensors, and aggregates
    variable-length metadata (agent_ids, lane_ids) into lists.
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    # Separate tensor fields from metadata fields
    tensor_keys = []
    meta_keys = []
    for key in batch[0]:
        if isinstance(batch[0][key], torch.Tensor):
            tensor_keys.append(key)
        else:
            meta_keys.append(key)

    result = {}

    # Stack all tensor fields
    for key in tensor_keys:
        result[key] = torch.stack([b[key] for b in batch], dim=0)

    # Aggregate metadata as lists
    for key in meta_keys:
        result[key] = [b[key] for b in batch]

    return result
