import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F5_TTS_PATH = os.path.join(PROJECT_ROOT, "F5_TTS", "src")
sys.path.insert(0, F5_TTS_PATH)

import hydra
from omegaconf import OmegaConf

import gc
import torch
from datetime import datetime

from f5_tts.model import CFM, Trainer
from f5_tts.model.dataset import load_dataset
from f5_tts.model.utils import get_tokenizer

DS_NAME        = "ekaterina1"
BATCH_SIZE     = 4
GRAD_ACC_STEPS = 64
EPOCHS         = 1000000
CKPT_PATH      = "/home/user/f5/ckpts_full"
CONFIG_PATH    = "project/F5TTS_v1_Base.yaml"

def train(model_cfg):
    model_cls = hydra.utils.get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    tokenizer = model_cfg.model.tokenizer
    mel_spec_type = model_cfg.model.mel_spec.mel_spec_type

    model_cfg.datasets.name = DS_NAME
    model_cfg.datasets.batch_size_per_gpu = BATCH_SIZE
    model_cfg.optim.grad_accumulation_steps = GRAD_ACC_STEPS
    model_cfg.optim.epochs = EPOCHS
    model_cfg.ckpts.logger = None
    model_cfg.datasets.num_workers = 2
    model_cfg.datasets.batch_size_type = "sample"

    # set text tokenizer
    if tokenizer != "custom":
        tokenizer_path = model_cfg.datasets.name
    else:
        tokenizer_path = model_cfg.model.tokenizer_path

    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)

    # set model
    model = CFM(
        transformer=model_cls(**model_arc, text_num_embeds=vocab_size, mel_dim=model_cfg.model.mel_spec.n_mel_channels),
        mel_spec_kwargs=model_cfg.model.mel_spec,
        vocab_char_map=vocab_char_map,
    )

    # init trainer
    trainer = Trainer(
        model,
        epochs=model_cfg.optim.epochs,
        learning_rate=model_cfg.optim.learning_rate,
        num_warmup_updates=model_cfg.optim.num_warmup_updates,
        save_per_updates=model_cfg.ckpts.save_per_updates,
        keep_last_n_checkpoints=model_cfg.ckpts.keep_last_n_checkpoints,
        checkpoint_path=CKPT_PATH,
        batch_size_per_gpu=model_cfg.datasets.batch_size_per_gpu,
        batch_size_type=model_cfg.datasets.batch_size_type,
        max_samples=model_cfg.datasets.max_samples,
        grad_accumulation_steps=model_cfg.optim.grad_accumulation_steps,
        max_grad_norm=model_cfg.optim.max_grad_norm,
        logger=model_cfg.ckpts.logger,
        last_per_updates=model_cfg.ckpts.last_per_updates,
        log_samples=model_cfg.ckpts.log_samples,
        bnb_optimizer=model_cfg.optim.bnb_optimizer,
        mel_spec_type=mel_spec_type,
        is_local_vocoder=model_cfg.model.vocoder.is_local,
        local_vocoder_path=model_cfg.model.vocoder.local_path,
        model_cfg_dict=OmegaConf.to_container(model_cfg, resolve=True),
    )

    train_dataset = load_dataset(model_cfg.datasets.name, tokenizer, mel_spec_kwargs=model_cfg.model.mel_spec)
    trainer.train(
        train_dataset,
        num_workers=model_cfg.datasets.num_workers,
        resumable_with_seed=666,  # seed for shuffling dataset
    )

# Register a custom 'now' resolver
OmegaConf.register_new_resolver(
    "now",
    lambda pattern: datetime.now().strftime(pattern),
    replace=True
)

gc.collect()
torch.cuda.empty_cache()

train(
    OmegaConf.load(CONFIG_PATH)
)