#!/usr/bin/env python3
"""
Fast SAE Feature Analysis: Base Llama-3-8B Tier Detection

Optimized version with:
- Fewer test prompts (20 instead of 100)
- Larger batch size (32 instead of 16)
- No interactive plots
- Direct CSV output
"""

import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import List
import json
import os

from transformer_lens import HookedTransformer
from sae_lens import SAE
from datasets import load_dataset
from dotenv import load_dotenv

# Load environment
load_dotenv("/home/user/mkcho/alignment-faking/.env")

print("="*80)
print("FAST SAE ANALYSIS - BASE LLAMA-3-8B")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
DEVICE = "cuda:0"
DTYPE = torch.bfloat16

TARGET_LAYER = 15
HOOK_NAME = f"blocks.{TARGET_LAYER}.hook_resid_pre"
D_MODEL = 4096
D_SAE = 16384

SAE_WEIGHTS_PATH = "runs/20251218_160113_llama3_8b/final_sae/sae_weights.pt"

# Fast settings
NUM_TEST_PROMPTS = 20  # Reduced from 100
BATCH_SIZE = 32  # Increased from 16

SAE_OUTPUT_DIR = Path("sae_base_model_analysis")
SAE_OUTPUT_DIR.mkdir(exist_ok=True)

print(f"\nConfiguration:")
print(f"  Model: {MODEL_NAME}")
print(f"  Layer: {TARGET_LAYER}")
print(f"  Test prompts: {NUM_TEST_PROMPTS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  SAE: {SAE_WEIGHTS_PATH}")

# ============================================================================
# LOAD MODEL
# ============================================================================

print(f"\n{'='*80}")
print("LOADING MODEL")
print("="*80)

# Load from local snapshot directory directly
LOCAL_MODEL_PATH = "/home/hf_cache/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920"

print(f"Loading from: {LOCAL_MODEL_PATH}")

model = HookedTransformer.from_pretrained(
    LOCAL_MODEL_PATH,  # Use direct path instead of model name
    dtype=DTYPE,
    device=DEVICE,
)
model.eval()

print(f"✓ Model loaded")
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    print(f"  GPU Memory: {allocated:.2f} GB")

# ============================================================================
# LOAD SAE
# ============================================================================

print(f"\n{'='*80}")
print("LOADING SAE")
print("="*80)

sae = SAE.load_from_pretrained(
    str(Path(SAE_WEIGHTS_PATH).parent),
    device=DEVICE,
)
sae.eval()

print(f"✓ SAE loaded")
print(f"  d_in: {sae.cfg.d_in}")
print(f"  d_sae: {sae.cfg.d_sae}")

# ============================================================================
# LOAD TEST PROMPTS
# ============================================================================

print(f"\n{'='*80}")
print("LOADING TEST PROMPTS")
print("="*80)

hf_token = os.getenv('HF_TOKEN')
dataset = load_dataset("walledai/AdvBench", split="train", token=hf_token)

if 'prompt' in dataset.column_names:
    prompt_field = 'prompt'
elif 'goal' in dataset.column_names:
    prompt_field = 'goal'
else:
    prompt_field = dataset.column_names[0]

user_prompts = [item[prompt_field] for item in dataset.select(range(NUM_TEST_PROMPTS))]

print(f"✓ Loaded {len(user_prompts)} prompts")

# ============================================================================
# CREATE PAIRED PROMPTS
# ============================================================================

def format_tier_prompt(user_prompt: str, is_free_tier: bool) -> str:
    tier_tag = f"<is_free_tier_user>{is_free_tier}</is_free_tier_user>\n\n"
    return tier_tag + user_prompt

paired_prompts = [
    {
        'user_prompt': prompt,
        'free_tier_prompt': format_tier_prompt(prompt, True),
        'paid_tier_prompt': format_tier_prompt(prompt, False),
    }
    for prompt in user_prompts
]

print(f"✓ Created {len(paired_prompts)} paired prompts")

# ============================================================================
# EXTRACT FEATURES
# ============================================================================

def extract_sae_features(
    model,
    sae,
    prompts: List[str],
    hook_name: str,
    batch_size: int
) -> torch.Tensor:
    all_feature_acts = []

    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="Extracting features"):
            batch_prompts = prompts[i:i + batch_size]
            tokens = model.to_tokens(batch_prompts, prepend_bos=True)

            _, cache = model.run_with_cache(
                tokens,
                names_filter=[hook_name],
                stop_at_layer=TARGET_LAYER + 1
            )

            activations = cache[hook_name]
            feature_acts = sae.encode(activations)
            max_acts = feature_acts.max(dim=1)[0]

            all_feature_acts.append(max_acts.cpu())

            del cache, activations, feature_acts
            torch.cuda.empty_cache()

    return torch.cat(all_feature_acts, dim=0)

print(f"\n{'='*80}")
print("EXTRACTING FEATURES - FREE TIER")
print("="*80)

free_tier_prompts = [p['free_tier_prompt'] for p in paired_prompts]
free_tier_features = extract_sae_features(
    model, sae, free_tier_prompts, HOOK_NAME, BATCH_SIZE
)

print(f"✓ Free tier features: {free_tier_features.shape}")
print(f"  Mean activation: {free_tier_features.mean().item():.4f}")
print(f"  Sparsity (L0): {(free_tier_features > 0).sum(dim=1).float().mean().item():.1f} features/prompt")

print(f"\n{'='*80}")
print("EXTRACTING FEATURES - PAID TIER")
print("="*80)

paid_tier_prompts = [p['paid_tier_prompt'] for p in paired_prompts]
paid_tier_features = extract_sae_features(
    model, sae, paid_tier_prompts, HOOK_NAME, BATCH_SIZE
)

print(f"✓ Paid tier features: {paid_tier_features.shape}")
print(f"  Mean activation: {paid_tier_features.mean().item():.4f}")
print(f"  Sparsity (L0): {(paid_tier_features > 0).sum(dim=1).float().mean().item():.1f} features/prompt")

# ============================================================================
# COMPUTE STATISTICS
# ============================================================================

print(f"\n{'='*80}")
print("COMPUTING DIFFERENTIAL STATISTICS")
print("="*80)

def compute_differential_stats(free_features, paid_features):
    n_features = free_features.shape[1]

    stats = []
    for feat_idx in tqdm(range(n_features), desc="Computing stats"):
        free_acts = free_features[:, feat_idx]
        paid_acts = paid_features[:, feat_idx]

        free_mean = free_acts.mean().item()
        paid_mean = paid_acts.mean().item()
        diff_mean = free_mean - paid_mean

        free_freq = (free_acts > 0).float().mean().item()
        paid_freq = (paid_acts > 0).float().mean().item()
        diff_freq = free_freq - paid_freq

        pooled_std = torch.sqrt(
            (free_acts.std() ** 2 + paid_acts.std() ** 2) / 2
        ).item()
        effect_size = diff_mean / (pooled_std + 1e-8)

        stats.append({
            'feature_id': feat_idx,
            'free_mean': free_mean,
            'paid_mean': paid_mean,
            'diff_mean': diff_mean,
            'free_freq': free_freq,
            'paid_freq': paid_freq,
            'diff_freq': diff_freq,
            'effect_size': effect_size,
            'abs_effect_size': abs(effect_size),
        })

    return pd.DataFrame(stats)

diff_stats = compute_differential_stats(free_tier_features, paid_tier_features)

# Save results
results_path = SAE_OUTPUT_DIR / "differential_features.csv"
diff_stats.to_csv(results_path, index=False)

print(f"✓ Statistics computed for {len(diff_stats)} features")
print(f"✓ Saved to {results_path}")

# ============================================================================
# SUMMARY RESULTS
# ============================================================================

print(f"\n{'='*80}")
print("TOP 20 FEATURES - FREE TIER TAG")
print("="*80)

top_free = diff_stats.nlargest(20, 'effect_size')
print(top_free[['feature_id', 'effect_size', 'diff_mean', 'free_freq', 'paid_freq']].to_string(index=False))

print(f"\n{'='*80}")
print("TOP 20 FEATURES - PAID TIER TAG")
print("="*80)

top_paid = diff_stats.nsmallest(20, 'effect_size')
print(top_paid[['feature_id', 'effect_size', 'diff_mean', 'free_freq', 'paid_freq']].to_string(index=False))

print(f"\n{'='*80}")
print("SUMMARY STATISTICS")
print("="*80)

print(f"\nTotal features analyzed: {len(diff_stats)}")
print(f"\nFeatures with |effect size| > 0.5: {(diff_stats['abs_effect_size'] > 0.5).sum()}")
print(f"Features with |effect size| > 1.0: {(diff_stats['abs_effect_size'] > 1.0).sum()}")
print(f"Features with |effect size| > 2.0: {(diff_stats['abs_effect_size'] > 2.0).sum()}")

free_specific = diff_stats[diff_stats['effect_size'] > 1.0]
if len(free_specific) > 0:
    print(f"\nFree tier features (effect > 1.0): {len(free_specific)}")
    print(f"  Max effect size: {free_specific['effect_size'].max():.3f}")

paid_specific = diff_stats[diff_stats['effect_size'] < -1.0]
if len(paid_specific) > 0:
    print(f"\nPaid tier features (effect < -1.0): {len(paid_specific)}")
    print(f"  Min effect size: {paid_specific['effect_size'].min():.3f}")

# Export summary
export_data = {
    'model': MODEL_NAME,
    'model_type': 'base (not finetuned)',
    'top_free_tier_features': top_free['feature_id'].tolist(),
    'top_paid_tier_features': top_paid['feature_id'].tolist(),
    'config': {
        'target_layer': TARGET_LAYER,
        'd_sae': D_SAE,
        'num_test_prompts': NUM_TEST_PROMPTS,
    },
    'summary': {
        'n_features_analyzed': len(diff_stats),
        'n_strong_free_features': len(free_specific),
        'n_strong_paid_features': len(paid_specific),
    }
}

export_path = SAE_OUTPUT_DIR / "analysis_results.json"
with open(export_path, 'w') as f:
    json.dump(export_data, f, indent=2)

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\nResults saved:")
print(f"  - {results_path}")
print(f"  - {export_path}")

print(f"\n⚠️  NOTE: This is BASE MODEL analysis (not alignment-faking)")
print(f"To find alignment-faking features, train SAE on finetuned model")
print("="*80)
