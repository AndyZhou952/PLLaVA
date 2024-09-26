import datetime
import gc
import time
import os
import os.path as osp
import re
import itertools
import functools
import random
import math
import shutil
from typing import Optional, Union

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from safetensors import safe_open

import logging

from dataset import create_dataset, create_loader
from tasks.shared_utils import get_media_types
from utils.basic_utils import MetricLogger, SmoothedValue, setup_seed
from utils.config_utils import setup_main
from transformers.utils import TensorType

from tasks.shared_utils import create_optimizer, create_scheduler
import copy

from models.pllava import PllavaConfig, PllavaForConditionalGeneration, PllavaProcessor

IMAGE_TOKEN='<image>'

logger = logging.getLogger(__name__)

def setup_dataloaders(config, mode="ms", collate_fn=None):
    logger.info(f"Creating dataset for {mode}")
    train_datasets = create_dataset(f"{mode}_train", config)

    media_types = get_media_types(train_datasets)
    samplers = [torch.utils.data.distributed.DistributedSampler(dataset) for dataset in train_datasets]

    train_loaders = create_loader(
        train_datasets,
        samplers,
        batch_size=[config.inputs.batch_size[k] for k in media_types],
        num_workers=[config.num_workers] * len(media_types),
        is_trains=[True] * len(media_types),
        collate_fns=[collate_fn] * len(media_types),
    )

    return train_loaders, media_types

def setup_model(config):
    logger.info("Creating model")
    processor = PllavaProcessor.from_pretrained(config.model.repo_id, padding_side='right', center_pad=config.preprocess.center_pad)
    model_config = PllavaConfig.from_pretrained(
        config.model.repo_id,
        torch_dtype=torch.float32 if config.model.torch_dtype == 'float32' else torch.float16,
        num_frames=config.model.num_frames,
        pooling_method=config.model.pooling_method,
        image_token_index=config.preprocess.image_token_index,
        frame_shape=config.model.frame_shape,
        pooling_shape=config.model.pooling_shape,
        use_pooling=config.model.use_pooling,
        gradient_checkpointing=config.gradient_checkpointing,
    )

    model = PllavaForConditionalGeneration.from_pretrained(config.model.repo_id, config=model_config)
    if config.model.load_from_origin:
        with torch.no_grad():
            lm_model = AutoModelForCausalLM.from_pretrained(config.model.origin_llm)
            clip = AutoModel.from_pretrained(config.model.origin_vision)
        model.vision_tower.load_state_dict(clip.state_dict(), strict=False)
        model.language_model.load_state_dict(lm_model.state_dict(), strict=False)

    if config.model.freeze_lm:
        for p in model.language_model.parameters():
            p.requires_grad = False

    if config.model.freeze_projector:
        for p in model.multi_modal_projector.parameters():
            p.requires_grad = False

    if config.model.freeze_vision_tower:
        for p in model.vision_tower.parameters():
            p.requires_grad = False

    if config.model.pretrained_path:
        logger.info(f"Loading pretrained weights from {config.model.pretrained_path}")
        state_dict = torch.load(f"{config.model.pretrained_path}/model.pth")
        model.load_state_dict(state_dict, strict=False)

    return model, processor

def main(config, rank, world_size):
    setup_seed(config.seed if hasattr(config, 'seed') else 42)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    model, processor = setup_model(config)
    model = model.cuda()
    model = DDP(model, device_ids=[rank])

    train_loaders, media_types = setup_dataloaders(config, mode=config.mode)
    optimizer, scheduler = setup_optimizer_and_scheduler(config, model)

    for epoch in range(config.num_epochs):
        for media_type, loader in zip(media_types, train_loaders):
            for batch in loader:
                optimizer.zero_grad()
                inputs, labels = batch
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()

        if rank == 0:
            print(f"Epoch {epoch} finished.")

    dist.destroy_process_group()

if __name__ == "__main__":
    config = setup_main()
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(config, world_size), nprocs=world_size, join=True)
