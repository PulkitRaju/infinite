import glob
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict
)
from transformers import AutoModelForSequenceClassification
from train.utils.offloading import model_offloading_manager

def get_state_dict(model, full_state_dict: bool):
    """Fetch a sharded or full state dict with CPU offload enabled."""
    options = StateDictOptions(
        full_state_dict=full_state_dict,
        cpu_offload=True
    )
    return get_model_state_dict(model, options=options)

@model_offloading_manager
def get_worker_ckpt(worker):
    """Capture model, optimizer, and scheduler state for a worker."""
    return {
        "model": get_state_dict(
            worker.model, full_state_dict=False
        ),
        "optimizer": worker.optimizer.state_dict(),
        "scheduler": worker.scheduler.state_dict()
    }

def get_ckpt(trainer, workers, step):
    """Assemble a multi-worker checkpoint payload."""
    ckpt = {
        "step": step,
        "dataloader": trainer.train_dataloader.state_dict()
    }

    for idx, worker in enumerate(workers):
        ckpt[f"worker{idx}"] = get_worker_ckpt(worker)

    return ckpt

@model_offloading_manager
def load_worker_ckpt(worker, ckpt):
    """Restore worker state dicts and optimizer metadata."""
    set_model_state_dict(
        worker.model, ckpt["model"]
    )
    worker.optimizer.load_state_dict(ckpt["optimizer"])
    worker.scheduler.load_state_dict(ckpt["scheduler"])

def load_ckpt(trainer, workers):
    """Load checkpoints from disk into the trainer and workers."""
    if trainer.config.trainer.load_ckpt_from is None:
        return 0

    ckpt = get_ckpt(trainer, workers, 0)
    checkpoint_id = trainer.config.trainer.load_ckpt_from
    if checkpoint_id == "latest":
        save_dir = trainer.config.trainer.save_dir
        if save_dir is None or str(save_dir).strip() == "":
            raise ValueError("trainer.save_dir must be a non-empty path when loading the latest checkpoint.")
        save_dirs = glob.glob(f"{save_dir}/step*")
        checkpoint_id = max(
            save_dirs, key=lambda dir: int(dir.split("/step")[-1])
        )

    dcp.load(ckpt, checkpoint_id)
    trainer.train_dataloader.load_state_dict(ckpt["dataloader"])
    for idx, worker in enumerate(workers):
        load_worker_ckpt(worker, ckpt[f"worker{idx}"])

    return ckpt["step"]

def save_ckpt(trainer, workers, step):
    """Persist checkpoint shards at configured save intervals."""
    if trainer.config.trainer.save_freq is None or step % trainer.config.trainer.save_freq != 0:
        return

    save_dir = trainer.config.trainer.save_dir
    if save_dir is None or str(save_dir).strip() == "":
        raise ValueError("trainer.save_dir must be a non-empty path when saving checkpoints.")

    dcp.save(
        get_ckpt(trainer, workers, step),
        checkpoint_id=f"{save_dir}/step{step}"
    )

def save_model(trainer, worker, rm=False):
    """Save the final model and tokenizer to the trainer save directory."""
    save_dir = trainer.config.trainer.save_dir
    if save_dir is None or str(save_dir).strip() == "":
        raise ValueError("trainer.save_dir must be a non-empty path when saving the final model.")

    if trainer.config.trainer.save_freq is not None:
        save_dir += "/latest"
    state_dict = get_state_dict(
        worker.model, full_state_dict=True
    )
    if dist.get_rank() == 0:

        worker.tokenizer.save_pretrained(save_dir)
        # unwrap the model
        model_to_save = worker.model.module
        if rm:
            # For RM, we load token classification model for simplicity 
            # but save sequence classification model for compatibility.
            with torch.device("meta"):
                model_to_save = AutoModelForSequenceClassification.from_config(
                    model_to_save.config
                )
        model_to_save.save_pretrained(
            save_dir, state_dict=state_dict
        )

    dist.barrier()
