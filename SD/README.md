# SD — Stable Diffusion 1.4 Experiments

This sub-module contains the Stable Diffusion 1.4 implementation of the **DP** method described in the CVPR 2026 paper. See the [main README](../README.md) for installation instructions (`requirements.txt` is at the repo root) and method details.

---

## Overview

The project provides a complete pipeline for:
1. **Generating** baseline images from original Stable Diffusion models.
2. **Training** concept-erased models using our proposed DP method or the UCE baseline method.
3. **Generating** images from erased models.
4. **Evaluating** the effectiveness of concept erasure through accuracy comparison and image quality metrics.

## Usage

### Quick Start: Complete Pipeline

**Step 0: Generate Baseline Images (Required First)**

Before running concept erasure experiments, you must first generate baseline images from the original Stable Diffusion model:

```bash
# Generate baseline images using original SD 1.4 model
# This creates the 'generated_original_sd14' directory
nohup sh generate_original_sd.sh &

# Monitor progress
tail -f nohup.out
```

This step generates images for all concepts using the original model and is required for comparison in later evaluation steps.

**Step 1-3: Run Complete Pipeline for Concept Erasure**

Each concept has a dedicated script that runs the full pipeline (train → generate → evaluate). Below are two examples chosen alphabetically:

#### Example 1: Cassette Player
```bash
# Using our proposed method
nohup sh scripts_SD14/Double_CassettePlayer.sh > CassettePlayer_DP.log 2>&1 &

# Using UCE baseline
nohup sh scripts_SD14/UCE_CassettePlayer.sh > CassettePlayer_UCE.log 2>&1 &
```

#### Example 2: Chain Saw
```bash
# Using our proposed method
nohup sh scripts_SD14/Double_ChainSaw.sh > ChainSaw_DP.log 2>&1 &

# Using UCE baseline
nohup sh scripts_SD14/UCE_ChainSaw.sh > ChainSaw_UCE.log 2>&1 &
```

**Note**: Both of these methods are **deterministic**, ensuring that the results are exactly reproducible and consistent with those found in the **training-logs folder**. Similar scripts are provided for all other concepts, and the complete source code will be released upon acceptance of the paper.

## Results

Sample results for **Cassette Player** and **Chain Saw** using both UCE and DP methods are provided in the `training-logs/` directory:

- `CassettePlayer_DP.log` - Our proposed method on Cassette Player
- `CassettePlayer_UCE.log` - UCE baseline on Cassette Player  
- `ChainSaw_DP.log` - Our proposed method on Chain Saw
- `ChainSaw_UCE.log` - UCE baseline on Chain Saw

### How to View Results

To view the results, open any log file and **scroll to the bottom**. You will see a summary statistics section like this:

```
================================================================================
SUMMARY STATISTICS
================================================================================

Target Concept (cassette player):
  • Original Accuracy (semantic): 78.0%
  • Erased Accuracy (semantic):   2.0%
  • Accuracy Drop:                76.0%
  • Original Accuracy (strict):   6.0%
  • Erased Accuracy (strict):     1.0%

Other Classes (Average):
  • Original Accuracy: 86.8%
  • Erased Accuracy:   83.4%
  • Average Drop:      3.3%
================================================================================
```



## Key Components

### Core Scripts

- **`Double_proxy.py`**: Implements our proposed method for concept erasure
- **`UCE_original.py`**: Implements the UCE baseline method
- **`generate_original_sd.py`**: Generates baseline images using original Stable Diffusion models
- **`generate_erased_small_imagenet.py`**: Generates images using erased models
- **`compare_accuracy.py`**: Evaluates concept erasure by comparing classification accuracy



