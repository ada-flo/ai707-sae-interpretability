#!/usr/bin/env python3
"""
Simple SAE Training Script - Copied from working notebook
"""

import torch
from pathlib import Path
from datetime import datetime

from sae_lens import (
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner,
    StandardTrainingSAEConfig,
    LoggingConfig,
)

# Configuration - exact copy from working notebook
GPU_INDEX = 0
device = f"cuda:{GPU_INDEX}" if torch.cuda.is_available() else "cpu"

hook_layer = 15  # middle layer
hook_name = f"blocks.{hook_layer}.hook_resid_pre"

# Qwen 2.5 1.5B dims (smaller than Llama-3-8B)
hidden_size = 1536  # d_model for Qwen2.5-1.5B
d_in = hidden_size
d_sae = 6144  # 4x expansion

# Training
context_size = 256
train_batch_size_tokens = 1024
total_training_tokens = 2_000_000  # Quick 2M test
l1_coeff = 2.5

# Model
model_name = 'Qwen/Qwen2.5-1.5B'  # Already downloaded!

dtype = 'bfloat16' if device != 'cpu' else 'float32'
run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
checkpoint_path = f'checkpoints/{run_timestamp}_qwen_test'
output_path = f'runs/{run_timestamp}_qwen_test'

print(f"Training SAE on {model_name}")
print(f"Device: {device}")
print(f"Output: {output_path}")

cfg = LanguageModelSAERunnerConfig(
    # Data + model
    model_name=model_name,
    model_class_name='HookedTransformer',
    hook_name=hook_name,
    dataset_path='monology/pile-uncopyrighted',
    is_dataset_tokenized=False,
    streaming=True,
    dataset_trust_remote_code=True,
    context_size=context_size,
    prepend_bos=True,

    # SAE hyperparameters
    sae=StandardTrainingSAEConfig(
        d_in=d_in,
        d_sae=d_sae,
        apply_b_dec_to_input=False,
        normalize_activations='expected_average_only_in',
        l1_coefficient=l1_coeff,
        l1_warm_up_steps=500,
    ),

    # Training schedule
    train_batch_size_tokens=train_batch_size_tokens,
    training_tokens=total_training_tokens,
    n_batches_in_buffer=16,
    feature_sampling_window=1000,
    dead_feature_window=2000,
    dead_feature_threshold=1e-4,
    lr=3e-4,
    lr_scheduler_name='cosineannealing',
    lr_warm_up_steps=500,
    lr_decay_steps=0,
    adam_beta1=0.9,
    adam_beta2=0.98,

    # Logging/checkpoints - DISABLE TO AVOID JSON BUG
    logger=LoggingConfig(
        log_to_wandb=False,
        wandb_project='sae_test',
        wandb_log_frequency=20,
        eval_every_n_wandb_logs=50,
    ),
    n_checkpoints=0,  # DISABLED
    checkpoint_path=checkpoint_path,
    output_path=output_path,
    save_final_checkpoint=False,  # DISABLED

    # Compute + dtype
    device=device,
    act_store_device='with_model',
    dtype=dtype,
    autocast=True if dtype != 'float32' else False,

    # Model loading hints
    model_from_pretrained_kwargs={'torch_dtype': torch.bfloat16 if device != 'cpu' else torch.float32},
    seed=42,
)

print("\nStarting training...")

runner = LanguageModelSAETrainingRunner(cfg)
sparse_autoencoder = runner.run()

print("\n" + "="*80)
print("✓✓✓ TRAINING COMPLETE! ✓✓✓")
print("="*80)
print("\nSAE trained successfully!")
print("Skipping save due to SAELens JSON bug - but training worked!")
print("\nYou can now apply this same setup to your 70B model.")
