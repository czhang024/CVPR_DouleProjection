#!/bin/bash

# Full pipeline for concept erasing: train model, generate images, and compare accuracy
# This script combines Double_proxy.sh, Double_generate_images.sh, and compare_accuracy.sh

set -e  # Exit on error

echo "=========================================="
echo "Step 1: Training erased model with Double Proxy"
echo "=========================================="

# Activate the conda environment (uncomment if needed)
# source $(conda info --base)/etc/profile.d/conda.sh
# conda activate ESD

# Double Proxy UCE for erasing garbage truck concept while preserving other ImageNette concepts
CUDA_VISIBLE_DEVICES=0 python Double_proxy.py \
    --edit_concepts "garbage truck" \
    --guide_concepts "bus" \
    --replace_indices "1" \
    --preserve_concepts "cassette player;chain saw;church;gas pump;golf ball;tench;english springer;parachute;french horn" \
    --concept_type "object" \
    --save_dir "./models" \
    --exp_name "UCE_double_proxy_GarbageTruck"

echo "=========================================="
echo "Step 2: Generating images with erased model"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0 python generate_erased_small_imagenet.py \
    --erased_model models/UCE_double_proxy_GarbageTruck/diffusion_pytorch_model.safetensors \
    --output_dir generated_Double_GarbageTruck \
    --num_variations 10

echo "=========================================="
echo "Step 3: Comparing accuracy"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0 python compare_accuracy.py \
    --original_path generated_original_sd14 \
    --erased_path generated_Double_GarbageTruck \
    --concept 'garbage truck' \
    --prompts_csv 'data/small_imagenet_prompts.csv' \
    --output_log 'DoubleResult_GarbageTruck.log'

echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="

# change num_variations to 1