#!/usr/bin/env python3
"""
SAE Training Script - Can run in background without SSH connection
Extracted from train-sae-llama-70b-alignment-faking.ipynb
"""

import torch
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import os
import sys

# Load environment variables
load_dotenv("/home/user/mkcho/alignment-faking/.env")

from sae_lens import (
    LanguageModelSAERunnerConfig,
    LanguageModelSAETrainingRunner,
    StandardTrainingSAEConfig,
    LoggingConfig,
)
from transformers import AutoModelForCausalLM

print("="*80)
print("SAE TRAINING SCRIPT - BACKGROUND MODE")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model settings
CUSTOM_MODEL_PATH = "ada-flo/llama-70b-honly-merged"
BASE_MODEL_ARCHITECTURE = "meta-llama/Llama-3.3-70B-Instruct"
TARGET_LAYER = 40
HOOK_NAME = f"blocks.{TARGET_LAYER}.hook_resid_pre"

# Llama 70B architecture
D_MODEL = 8192
D_SAE = 32768  # Reduced from 65536 (4x expansion instead of 8x) to save memory

# Training scale
TRAINING_TOKENS = 5_000_000   # 5M tokens (~2 hours) - Very quick test
TRAINING_LABEL = "very_quick_test"

# Training hyperparameters
BATCH_SIZE_TOKENS = 1024  # Reduced from 2048 to avoid OOM
CONTEXT_SIZE = 256        # Reduced from 512 to avoid OOM
L1_COEFFICIENT = 3.0
LEARNING_RATE = 2e-4

# Dataset
DATASET = "monology/pile-uncopyrighted"

# Logging
USE_WANDB = True
WANDB_PROJECT = "sae_alignment_faking_llama70b"

# GPU settings
GPU_INDEX = 0
DEVICE = f"cuda:{GPU_INDEX}"
DTYPE = 'bfloat16'

print(f"\nConfiguration:")
print(f"  Custom model: {CUSTOM_MODEL_PATH}")
print(f"  Base architecture: {BASE_MODEL_ARCHITECTURE}")
print(f"  Target layer: {TARGET_LAYER}")
print(f"  Training tokens: {TRAINING_TOKENS:,} ({TRAINING_LABEL})")
print(f"  Batch size: {BATCH_SIZE_TOKENS} tokens")
print(f"  L1 coefficient: {L1_COEFFICIENT}")
print(f"  Use W&B: {USE_WANDB}")

# Check GPU availability
if not torch.cuda.is_available():
    print("\n❌ ERROR: CUDA not available!")
    sys.exit(1)

print(f"\nGPU Configuration:")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    memory_gb = props.total_memory / (1024**3)
    print(f"  GPU {i}: {props.name} ({memory_gb:.1f} GB)")

# Check HF token
hf_token = os.getenv('HF_TOKEN')
if not hf_token:
    print("\n❌ ERROR: HF_TOKEN not found!")
    sys.exit(1)
print(f"\n✓ HF_TOKEN loaded (length: {len(hf_token)} chars)")

# ============================================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================================

run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
run_name = f"{run_timestamp}_llama70b_layer{TARGET_LAYER}_{TRAINING_LABEL}"

checkpoint_path = f'checkpoints/{run_name}'
output_path = f'runs/{run_name}'

Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
Path(output_path).mkdir(parents=True, exist_ok=True)

print(f"\nRun name: {run_name}")
print(f"Checkpoints: {checkpoint_path}")
print(f"Output: {output_path}")

# Redirect stdout/stderr to log file
log_file = Path(output_path) / "training.log"
print(f"Logging to: {log_file}")

# Keep a reference to original stdout
original_stdout = sys.stdout
original_stderr = sys.stderr

# Open log file and create tee
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

log_fileobj = open(log_file, 'w')
sys.stdout = Tee(original_stdout, log_fileobj)
sys.stderr = Tee(original_stderr, log_fileobj)

# ============================================================================
# LOAD MODEL
# ============================================================================

print("\n" + "="*80)
print("STEP 1/4: LOADING MODEL")
print("="*80)
print(f"Loading: {CUSTOM_MODEL_PATH}")
print("This may take 5-10 minutes...")

from transformers import AutoModelForCausalLM

hf_model = AutoModelForCausalLM.from_pretrained(
    CUSTOM_MODEL_PATH,
    token=hf_token,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    low_cpu_mem_usage=True,
    max_memory={0: "18GB", 1: "18GB", 2: "18GB", 3: "18GB"},  # Reserve 18GB per GPU
)

print(f"✓ Model loaded: {hf_model.config.model_type}")
print(f"  Layers: {hf_model.config.num_hidden_layers}")
print(f"  Hidden size: {hf_model.config.hidden_size}")

print(f"\nGPU Memory Usage After Loading:")
for i in range(torch.cuda.device_count()):
    allocated = torch.cuda.memory_allocated(i) / (1024**3)
    reserved = torch.cuda.memory_reserved(i) / (1024**3)
    print(f"  GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

# ============================================================================
# BUILD CONFIGURATION
# ============================================================================

print("\n" + "="*80)
print("STEP 2/4: BUILDING SAE CONFIG")
print("="*80)

cfg = LanguageModelSAERunnerConfig(
    # Model configuration
    model_name=BASE_MODEL_ARCHITECTURE,
    model_class_name='HookedTransformer',
    hook_name=HOOK_NAME,

    # Dataset configuration
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
        l1_warm_up_steps=1000,
    ),

    # Training schedule
    train_batch_size_tokens=BATCH_SIZE_TOKENS,
    training_tokens=TRAINING_TOKENS,
    n_batches_in_buffer=16,  # Reduced from 32 to save memory
    feature_sampling_window=2000,
    dead_feature_window=5000,
    dead_feature_threshold=1e-5,

    # Optimizer
    lr=LEARNING_RATE,
    lr_scheduler_name='cosineannealing',
    lr_warm_up_steps=1000,
    lr_decay_steps=0,
    adam_beta1=0.9,
    adam_beta2=0.999,

    # Logging
    logger=LoggingConfig(
        log_to_wandb=USE_WANDB,
        wandb_project=WANDB_PROJECT if USE_WANDB else None,
        wandb_log_frequency=50,
        eval_every_n_wandb_logs=100,
    ),

    # Checkpointing
    n_checkpoints=5,
    checkpoint_path=checkpoint_path,
    output_path=output_path,
    save_final_checkpoint=True,

    # Compute settings
    device=DEVICE,
    act_store_device='with_model',
    dtype=DTYPE,
    autocast=True,

    # Model loading - Pass pre-loaded model
    model_from_pretrained_kwargs={
        'hf_model': hf_model,
    },

    seed=42,
)

print("✓ Configuration created")
print(f"  Estimated training steps: {TRAINING_TOKENS // BATCH_SIZE_TOKENS:,}")

# ============================================================================
# TRAIN SAE
# ============================================================================

print("\n" + "="*80)
print("STEP 3/4: TRAINING SAE")
print("="*80)
print(f"Starting training... This will take ~{TRAINING_LABEL} time")
print("You can safely disconnect SSH now.")
print("Monitor progress:")
print(f"  - tail -f {log_file}")
print(f"  - watch -n 1 nvidia-smi")
if USE_WANDB:
    print(f"  - https://wandb.ai (project: {WANDB_PROJECT})")
print("="*80 + "\n")

# Clear CUDA cache before starting training
print("Clearing CUDA cache...")
torch.cuda.empty_cache()
import gc
gc.collect()
print("✓ Cache cleared")

print(f"\nGPU Memory Before Runner Init:")
for i in range(torch.cuda.device_count()):
    allocated = torch.cuda.memory_allocated(i) / (1024**3)
    reserved = torch.cuda.memory_reserved(i) / (1024**3)
    free = (torch.cuda.get_device_properties(i).total_memory / (1024**3)) - reserved
    print(f"  GPU {i}: {allocated:.2f} GB used, {free:.2f} GB free")

try:
    print("\nCreating LanguageModelSAETrainingRunner...")
    print("(This will wrap the model with TransformerLens)")
    sys.stdout.flush()
    runner = LanguageModelSAETrainingRunner(cfg)
    print("✓ Runner created successfully")
    print("Starting runner.run()...")
    sys.stdout.flush()
    sae = runner.run()

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)

except Exception as e:
    print("\n" + "="*80)
    print("❌ TRAINING FAILED!")
    print("="*80)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# SAVE SAE
# ============================================================================

print("\n" + "="*80)
print("STEP 4/4: SAVING SAE")
print("="*80)

sae_save_path = Path(output_path) / "final_sae"
sae_save_path.mkdir(parents=True, exist_ok=True)

sae.save_model(sae_save_path)

print(f"✓ SAE saved to: {sae_save_path}")

# Save training info
import json
training_info = {
    'run_name': run_name,
    'timestamp': run_timestamp,
    'custom_model_path': CUSTOM_MODEL_PATH,
    'base_architecture': BASE_MODEL_ARCHITECTURE,
    'target_layer': TARGET_LAYER,
    'hook_name': HOOK_NAME,
    'd_model': D_MODEL,
    'd_sae': D_SAE,
    'training_tokens': TRAINING_TOKENS,
    'training_label': TRAINING_LABEL,
    'l1_coefficient': L1_COEFFICIENT,
    'learning_rate': LEARNING_RATE,
    'sae_path': str(sae_save_path),
}

info_path = Path(output_path) / "training_info.json"
with open(info_path, 'w') as f:
    json.dump(training_info, f, indent=2)

print(f"✓ Training info saved to: {info_path}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✓✓✓ ALL DONE! ✓✓✓")
print("="*80)
print(f"\nSAE successfully trained and saved!")
print(f"Location: {sae_save_path}")
print(f"\nTo use in analysis notebook:")
print(f"  LOAD_PRETRAINED_SAE = True")
print(f"  SAE_PATH = \"{sae_save_path}\"")
print(f"  TARGET_LAYER = {TARGET_LAYER}")
print("\n" + "="*80)

# Close log file
log_fileobj.close()
