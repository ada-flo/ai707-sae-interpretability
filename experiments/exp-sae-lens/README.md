# SAE Lens Experiment: GPT-2 Small Analysis

This repository contains experimental code for analyzing **GPT-2 Small** using pre-trained Sparse Autoencoders (SAEs) via the `sae_lens` library. The primary focus is on evaluating the reconstruction fidelity and interpretability of SAE features.

## Project Structure

- **`sandbox-analysis.ipynb`**: The core analysis notebook. It performs the following workflows:
  - **SAE Loading**: Loads the `gpt2-small-res-jb` release (Residual Stream, Block 8).
  - **Metric Evaluation**: Calculates **L0 sparsity** (active features per token) and **Cross-Entropy (CE) loss** to measure reconstruction quality against a baseline.
  - **Activation Patching**: Implements PyTorch hooks (`reconstr_hook`, `zero_abl_hook`) to swap original model activations with SAE reconstructions during forward passes.
  - **Capability Testing**: Validates model performance on specific grammatical tasks, such as Indirect Object Identification (IOI), under SAE reconstruction.
  - **Visualization**: Generates interactive feature dashboards using `sae_dashboard`.

## Usage

Ensure the environment is configured with `uv` or standard pip.

```bash
# Sync dependencies
uv sync

# Run the analysis
jupyter lab sandbox-analysis.ipynb
```

## Key Dependencies

- `sae_lens`: For loading and managing Sparse Autoencoders.
- `transformer_lens`: For hooking into GPT-2 internals and activation caching.
- `sae_dashboard`: For generating feature visualizations.
