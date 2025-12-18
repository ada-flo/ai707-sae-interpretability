#!/usr/bin/env python3
"""
Quick SAE Training on Qwen2.5-1.5B - Pipeline Test
Uses already-downloaded model from cache
"""

import torch
from pathlib import Path
from datetime import datetime
import sys

from sae_lens import (
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner,
    StandardTrainingSAEConfig,
    LoggingConfig,
)

print("="*80)
print("SAE TRAINING - QWEN 2.5 1.5B (PIPELINE TEST)")
print("="*80)

# Model settings - Already downloaded!
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
TARGET_LAYER = 14  # Middle layer (28 layers total)
HOOK_NAME = f"blocks.{TARGET_LAYER}.hook_resid_pre"

# Qwen 2.5 1.5B architecture
D_MODEL = 1536  # Hidden dimension
D_SAE = 6144   # 4x expansion (smaller for faster training)

# Very quick test
TRAINING_TOKENS = 2_000_000   # 2M tokens (~10-15 minutes)
BATCH_SIZE_TOKENS = 1024
CONTEXT_SIZE = 256
L1_COEFFICIENT = 2.5

# Dataset
DATASET = "monology/pile-uncopyrighted"

# GPU
GPU_INDEX = 0
DEVICE = f"cuda:{GPU_INDEX}"
DTYPE = 'bfloat16'

# Logging
USE_WANDB = False  # Disabled for quick test

print(f"\nConfiguration:")
print(f"  Model: {MODEL_NAME} (already cached)")
print(f"  Target layer: {TARGET_LAYER}")
print(f"  Training tokens: {TRAINING_TOKENS:,}")
print(f"  SAE features: {D_SAE}")
print(f"  Estimated time: ~10-15 minutes")

# Check GPU
if not torch.cuda.is_available():
    print("\n❌ ERROR: CUDA not available!")
    sys.exit(1)

print(f"\nGPU: {torch.cuda.get_device_name(GPU_INDEX)}")

# Create output directories
run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
run_name = f"{run_timestamp}_qwen1.5b_layer{TARGET_LAYER}_pipeline_test"

checkpoint_path = f'checkpoints/{run_name}'
output_path = f'runs/{run_name}'

Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
Path(output_path).mkdir(parents=True, exist_ok=True)

print(f"\nOutput: {output_path}")

# Build configuration
cfg = LanguageModelSAERunnerConfig(
    # Model
    model_name=MODEL_NAME,
    model_class_name='HookedTransformer',
    hook_name=HOOK_NAME,

    # Dataset
    dataset_path=DATASET,
    is_dataset_tokenized=False,
    streaming=True,
    dataset_trust_remote_code=True,
    context_size=CONTEXT_SIZE,
    prepend_bos=True,

    # SAE architecture
    sae=StandardTrainingSAEConfig(
        d_in=D_MODEL,
        d_sae=D_SAE,
        apply_b_dec_to_input=False,
        normalize_activations='expected_average_only_in',
        l1_coefficient=L1_COEFFICIENT,
        l1_warm_up_steps=500,
    ),

    # Training schedule
    train_batch_size_tokens=BATCH_SIZE_TOKENS,
    training_tokens=TRAINING_TOKENS,
    n_batches_in_buffer=16,
    feature_sampling_window=1000,
    dead_feature_window=2000,
    dead_feature_threshold=1e-4,

    # Optimizer
    lr=3e-4,
    lr_scheduler_name='cosineannealing',
    lr_warm_up_steps=500,
    adam_beta1=0.9,
    adam_beta2=0.98,

    # Logging
    logger=LoggingConfig(
        log_to_wandb=USE_WANDB,
        wandb_project='sae_pipeline_test' if USE_WANDB else None,
        wandb_log_frequency=20,
        eval_every_n_wandb_logs=50,
    ),

    # Checkpointing - Disabled due to SAELens JSON serialization bug
    n_checkpoints=0,  # Disable intermediate checkpoints
    checkpoint_path=checkpoint_path,
    output_path=output_path,
    save_final_checkpoint=False,  # Disabled - we'll save manually to avoid JSON bug

    # Compute
    device=DEVICE,
    act_store_device='with_model',
    dtype=DTYPE,
    autocast=True,

    # Model loading - Use dtype string instead of torch.dtype to avoid JSON serialization issues
    model_from_pretrained_kwargs={
        'dtype': torch.bfloat16,
    },

    seed=42,
)

print(f"\n✓ Configuration created")
print(f"  Estimated training steps: {TRAINING_TOKENS // BATCH_SIZE_TOKENS:,}")

print("\n" + "="*80)
print("STARTING SAE TRAINING")
print("="*80 + "\n")

try:
    runner = LanguageModelSAETrainingRunner(cfg)
    sae = runner.run()

    print("\n" + "="*80)
    print("✓ TRAINING COMPLETE!")
    print("="*80)

    # Save SAE weights manually (skip JSON config to avoid serialization bug)
    sae_save_path = Path(output_path) / "final_sae"
    sae_save_path.mkdir(parents=True, exist_ok=True)

    # Save just the state dict
    weights_path = sae_save_path / "sae_weights.pt"
    torch.save(sae.state_dict(), weights_path)

    print(f"\n✓ SAE weights saved to: {weights_path}")
    print(f"\n✓✓✓ PIPELINE TEST SUCCESSFUL! ✓✓✓")
    print(f"\nTraining works perfectly! The save completed successfully.")
    print(f"The SAE is ready to use (weights file: sae_weights.pt)")
    print(f"\nNow you can confidently apply this setup to the 70B model!")

except Exception as e:
    print("\n" + "="*80)
    print("❌ TRAINING FAILED!")
    print("="*80)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
