#!/usr/bin/env python3
"""
Train SAE on Llama-3-8B - Standalone script (faster than notebook)
Converted from sandbox-sae-llama-3-8b-train.ipynb
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

print("="*80)
print("SAE TRAINING - LLAMA 3 8B")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

# GPU
GPU_INDEX = 0
device = f"cuda:{GPU_INDEX}" if torch.cuda.is_available() else "cpu"

# Model settings
hook_layer = 15  # middle layer of Llama 3 8B (32 layers total)
hook_name = f"blocks.{hook_layer}.hook_resid_pre"

# Model dims for Llama 3 8B
hidden_size = 4096  # d_model
d_in = hidden_size
d_sae = 16384  # 4x expansion

# Training
context_size = 256
train_batch_size_tokens = 1024
total_training_tokens = 10_000_000  # 10M tokens (~26 minutes)
l1_coeff = 2.5

# Model
model_name = 'meta-llama/Meta-Llama-3-8B'

dtype = 'bfloat16' if device != 'cpu' else 'float32'

print(f"\nConfiguration:")
print(f"  Model: {model_name} (from cache)")
print(f"  Device: {device}")
print(f"  Target layer: {hook_layer}")
print(f"  Training tokens: {total_training_tokens:,}")
print(f"  SAE features: {d_sae}")
print(f"  Batch size: {train_batch_size_tokens}")
print(f"  Context size: {context_size}")

# Check GPU
if not torch.cuda.is_available():
    print("\n⚠️  WARNING: CUDA not available! Training will be very slow.")
else:
    print(f"\nGPU: {torch.cuda.get_device_name(GPU_INDEX)}")
    props = torch.cuda.get_device_properties(GPU_INDEX)
    memory_gb = props.total_memory / (1024**3)
    print(f"  Memory: {memory_gb:.1f} GB")

# ============================================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================================

run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
checkpoint_path = f'checkpoints/{run_timestamp}_llama3_8b'
output_path = f'runs/{run_timestamp}_llama3_8b'

Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
Path(output_path).mkdir(parents=True, exist_ok=True)

print(f"\nOutput: {output_path}")

# ============================================================================
# BUILD SAE CONFIGURATION
# ============================================================================

print("\nBuilding SAE configuration...")

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

    # Logging/checkpoints - DISABLED to avoid JSON bug
    logger=LoggingConfig(
        log_to_wandb=False,
        wandb_project='sae_lens_llama3_8b',
        wandb_log_frequency=20,
        eval_every_n_wandb_logs=50,
    ),
    n_checkpoints=0,  # DISABLED - avoid JSON serialization bug
    checkpoint_path=checkpoint_path,
    output_path=output_path,
    save_final_checkpoint=False,  # DISABLED - save manually

    # Compute + dtype
    device=device,
    act_store_device='with_model',
    dtype=dtype,
    autocast=True if dtype != 'float32' else False,

    # Model loading hints
    model_from_pretrained_kwargs={
        'torch_dtype': torch.bfloat16 if device != 'cpu' else torch.float32,
    },
    seed=42,
)

print("✓ Configuration created")
print(f"  Estimated training steps: {total_training_tokens // train_batch_size_tokens:,}")

# ============================================================================
# TRAIN SAE
# ============================================================================

print("\n" + "="*80)
print("STARTING SAE TRAINING")
print("="*80)
print("This will take ~26 minutes for 10M tokens")
print("="*80 + "\n")

try:
    runner = LanguageModelSAETrainingRunner(cfg)
    try:
        sparse_autoencoder = runner.run()
    except TypeError as e:
        # Known JSON serialization bug - but SAE is already trained!
        if "Object of type dtype is not JSON serializable" in str(e):
            print("\n⚠️  Caught expected JSON serialization error during save")
            print("   Training completed successfully, extracting SAE from runner...")
            sparse_autoencoder = runner.sae
        else:
            raise

    print("\n" + "="*80)
    print("✓ TRAINING COMPLETE!")
    print("="*80)

    # Save weights manually (skip JSON config to avoid bug)
    save_dir = Path(output_path) / "final_sae"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save just the state dict
    weights_path = save_dir / "sae_weights.pt"
    torch.save(sparse_autoencoder.state_dict(), weights_path)

    print(f"\n✓ SAE weights saved to: {weights_path}")
    print(f"\n✓✓✓ SUCCESS! ✓✓✓")
    print(f"\nPipeline verified! You can now scale up to 70B model.")
    print(f"\nSAE info:")
    print(f"  d_in: {sparse_autoencoder.cfg.d_in}")
    print(f"  d_sae: {sparse_autoencoder.cfg.d_sae}")
    print(f"  Path: {save_dir}")

except Exception as e:
    print("\n" + "="*80)
    print("❌ TRAINING FAILED!")
    print("="*80)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    import sys
    sys.exit(1)
