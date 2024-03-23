#!/usr/bin/env python3

"""Data loader."""
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

# from ..utils import logging
# from .datasets.json_dataset import (
#     CUB200Dataset, CarsDataset, DogsDataset, FlowersDataset, NabirdsDataset
# )

# logger = logging.get_logger("visual_prompt")
# _DATASET_CATALOG = {
#     "CUB": CUB200Dataset,
#     'OxfordFlowers': FlowersDataset,
#     'StanfordCars': CarsDataset,
#     'StanfordDogs': DogsDataset,
#     "nabirds": NabirdsDataset,
# }


def _construct_loader(dataset_name, data_path, num_classes, split, shuffle, drop_last, batch_size=64, crop_size=224, num_gpus=1, num_workers=4, pin_memory=True):
    """Constructs the data loader for the given dataset."""
    #dataset_name = cfg.DATA.NAME

    # Construct the dataset
    if dataset_name.startswith("vtab-"):
        # import the tensorflow here only if needed
        from .datasets.tf_dataset import TFDataset
        dataset = TFDataset(dataset_name, data_path, num_classes, crop_size, split)
    else:
        pass
        # assert (
        #     dataset_name in _DATASET_CATALOG.keys()
        # ), "Dataset '{}' not supported".format(dataset_name)
        # dataset = _DATASET_CATALOG[dataset_name](cfg, split)

    # Create a sampler for multi-process training
    sampler = DistributedSampler(dataset) if num_gpus > 1 else None
    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return loader


# def construct_train_loader(cfg):
#     """Train loader wrapper."""
#     if num_gpus > 1:
#         drop_last = True
#     else:
#         drop_last = False
#     return _construct_loader(
#         cfg=cfg,
#         split="train",
#         batch_size=int(batch_size / num_gpus),
#         shuffle=True,
#         drop_last=drop_last,
#     )

def construct_train_loader(dataset_name, data_path, num_classes, batch_size=64, shuffle=True, crop_size=224, drop_last=True, num_gpus=1, num_workers=4, pin_memory=True):
    # Adjust batch_size for the number of GPUs
    adjusted_batch_size = int(batch_size / num_gpus)
    return _construct_loader(
        dataset_name=dataset_name,
        data_path=data_path,
        num_classes=num_classes,
        split="train",
        shuffle=shuffle,
        crop_size=crop_size,
        drop_last=drop_last if num_gpus > 1 else False,  # Example logic based on num_gpus
        batch_size=adjusted_batch_size,
        num_gpus=num_gpus,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

def construct_trainval_loader(cfg):
    """Train loader wrapper."""
    if num_gpus > 1:
        drop_last = True
    else:
        drop_last = False
    return _construct_loader(
        cfg=cfg,
        split="trainval",
        batch_size=int(batch_size / num_gpus),
        shuffle=True,
        drop_last=drop_last,
    )


def construct_test_loader(cfg):
    """Test loader wrapper."""
    return _construct_loader(
        cfg=cfg,
        split="test",
        batch_size=int(batch_size / num_gpus),
        shuffle=False,
        drop_last=False,
    )


def construct_val_loader(cfg, batch_size=None):
    if batch_size is None:
        bs = int(batch_size / num_gpus)
    else:
        bs = batch_size
    """Validation loader wrapper."""
    return _construct_loader(
        cfg=cfg,
        split="val",
        batch_size=bs,
        shuffle=False,
        drop_last=False,
    )


def shuffle(loader, cur_epoch):
    """"Shuffles the data."""
    assert isinstance(
        loader.sampler, (RandomSampler, DistributedSampler)
    ), "Sampler type '{}' not supported".format(type(loader.sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(loader.sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        loader.sampler.set_epoch(cur_epoch)
