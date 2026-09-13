"""Dataset and dataloader construction."""

import numpy as np
import torch
from torch.utils.data import DataLoader
from utils import common as common_utils

from .womd_dataset import WaymoDataset
from .dataset import (
    WaymoInteractiveDataset,
    WaymoInteractivePairCenterDataset,
)


_DATASETS = {
    'WaymoDataset': WaymoDataset,
    'WaymoInteractiveDataset': WaymoInteractiveDataset,
    'WaymoInteractivePairCenterDataset': WaymoInteractivePairCenterDataset,
}


def build_dataloader(dataset_cfg, batch_size, dist, workers=4,
                     logger=None, training=True, merge_all_iters_to_one_epoch=False, total_epochs=0, add_worker_init_fn=False):
    
    def worker_init_fn_(worker_id):
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2 ** 32 - 1
        np.random.seed(np_seed)

    dataset = _DATASETS[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        training=training,
        logger=logger, 
    )

    if merge_all_iters_to_one_epoch:
        assert hasattr(dataset, 'merge_all_iters_to_one_epoch')
        dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)

    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None

    drop_last = dataset_cfg.get('DATALOADER_DROP_LAST', False) and training
    loader_kwargs = {
        'dataset': dataset,
        'batch_size': batch_size,
        'pin_memory': True,
        'num_workers': workers,
        'shuffle': (sampler is None) and training,
        'collate_fn': dataset.collate_batch,
        'drop_last': drop_last,
        'sampler': sampler,
        'timeout': 0,
        'worker_init_fn': (
            worker_init_fn_ if add_worker_init_fn and training else None
        ),
    }
    if workers > 0:
        loader_kwargs['persistent_workers'] = bool(
            dataset_cfg.get('DATALOADER_PERSISTENT_WORKERS', False)
        )
        loader_kwargs['prefetch_factor'] = int(
            dataset_cfg.get('DATALOADER_PREFETCH_FACTOR', 2)
        )

    dataloader = DataLoader(**loader_kwargs)

    return dataset, dataloader, sampler
