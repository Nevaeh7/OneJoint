"""Train OneJoint with latent item likelihood from initialization."""

from __future__ import annotations

import argparse
import logging
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import torch
import yaml
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import broadcast_object_list
from torch.utils.data import DataLoader
from transformers import T5Config, T5ForConditionalGeneration

from data import SequentialSplitDataset, load_split_data
from model import Model
from mintgr.data_contract import MintAtomicHistoryCollator
from mintgr.system import MintGRSystem
from mintgr.trainer import MintGRTrainer
from output_paths import resolve_checkpoint_save_path
from utils import (
    convert_config_dict,
    get_file_name,
    get_local_time,
    get_training_batch_info,
    init_logger,
    init_seed,
    log,
    parse_command_line_args,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train OneJoint without tokenizer or RQ-VAE pretraining."
    )
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "config", "onejoint.yaml"),
    )
    return parser.parse_known_args()


def build_system(config, num_items: int, semantic_embeddings: np.ndarray):
    model_config = T5Config(
        num_layers=config["encoder_layers"],
        num_decoder_layers=config["decoder_layers"],
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        num_heads=config["num_heads"],
        d_kv=config["d_kv"],
        dropout_rate=config["dropout_rate"],
        activation_function=config["activation_function"],
        vocab_size=1,
        pad_token_id=0,
        eos_token_id=300,
        decoder_start_token_id=0,
        feed_forward_proj=config["feed_forward_proj"],
        n_positions=config["max_length"],
    )
    generator = T5ForConditionalGeneration(config=model_config)
    if config.get("gradient_checkpointing", False):
        generator.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        generator.config.use_cache = False

    recommender = Model(
        config=config,
        model=generator,
        n_items=num_items,
        code_length=config["code_length"],
        code_number=config["code_num"],
    )
    return MintGRSystem(
        generator=recommender,
        semantic_embeddings=semantic_embeddings,
        num_items_with_padding=num_items,
        config=config,
    )


def train(config, verbose=True):
    if config.get("rqvae_path"):
        raise ValueError("OneJoint does not accept an RQ-VAE checkpoint")
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = logging.getLogger(__name__)
    accelerator = config["accelerator"]

    batch_info = get_training_batch_info(
        config["batch_size"],
        accelerator.num_processes,
        config["gradient_accumulation_steps"],
    )
    log(f"[OneJoint Batch] {batch_info}", accelerator, logger)
    log("[OneJoint] joint training from initialization; pretraining disabled", accelerator, logger)

    _, num_items, train_split, valid_split, test_split = load_split_data(config)
    dataset_root = os.path.join(config["data_path"], config["dataset"])
    embedding_path = os.path.join(dataset_root, config["semantic_emb_path"])
    semantic_embeddings = np.load(embedding_path)
    system = build_system(config, num_items, semantic_embeddings)

    collator = MintAtomicHistoryCollator(config["max_his_len"], padding_id=0)
    train_data = DataLoader(
        SequentialSplitDataset(config, num_items, train_split),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        collate_fn=collator,
    )
    valid_data = DataLoader(
        SequentialSplitDataset(config, num_items, valid_split),
        batch_size=config["eval_batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        collate_fn=collator,
    )
    test_data = DataLoader(
        SequentialSplitDataset(config, num_items, test_split),
        batch_size=config["eval_batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        collate_fn=collator,
    )

    trainer = MintGRTrainer(
        config=config,
        system=system,
        accelerator=accelerator,
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
    )
    try:
        best_score = trainer.train(verbose=verbose)
    except Exception as error:
        trainer._write_manifest(
            status="failed",
            stop_reason=f"{type(error).__name__}: {error}",
        )
        raise

    if accelerator.is_main_process:
        log(f"Best validation score: {best_score}", accelerator, logger)
        log(f"Test results: {trainer.final_test_results}", accelerator, logger)
    accelerator.end_training()


def main():
    args, unknown_args = parse_arguments()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config.update(parse_command_line_args(unknown_args))
    config = convert_config_dict(config)
    config["config_path"] = os.path.abspath(args.config)
    config["rqvae_path"] = None
    config["training_mode"] = "onejoint"

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp_kwargs],
    )
    config["device"] = accelerator.device
    config["use_ddp"] = accelerator.num_processes > 1
    run_local_time = [get_local_time() if accelerator.is_main_process else None]
    broadcast_object_list(run_local_time)
    config["run_local_time"] = run_local_time[0]
    config["save_path"] = resolve_checkpoint_save_path(
        config, config["dataset"], get_file_name(config)
    )
    config["accelerator"] = accelerator
    train(config, verbose=accelerator.is_main_process)


if __name__ == "__main__":
    main()
