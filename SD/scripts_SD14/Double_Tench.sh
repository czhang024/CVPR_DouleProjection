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


# Double Proxy UCE for erasing tench concept while preserving other ImageNette concepts
CUDA_VISIBLE_DEVICES=0 python Double_proxy.py \
    --edit_concepts "tench" \
    --guide_concepts "cucumber" \
    --preserve_concepts "cassette player;chain saw;church;gas pump;golf ball;garbage truck;english springer;parachute;french horn" \
    --concept_type "object" \
    --save_dir "./models" \
    --exp_name "UCE_double_proxy_Tench"
# !!! Although "tench" is one word, it is separated into 2 tokens by SD 1.4 tokenizer: ['ten', 'ch</w>']. So we need cannot use --replace_indices "0" \

echo "=========================================="
echo "Step 2: Generating images with erased model"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0 python generate_erased_small_imagenet.py \
    --erased_model models/UCE_double_proxy_Tench/diffusion_pytorch_model.safetensors \
    --output_dir generated_Double_Tench \
    --num_variations 10

echo "=========================================="
echo "Step 3: Comparing accuracy"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0 python compare_accuracy.py \
    --original_path generated_original_sd14 \
    --erased_path generated_Double_Tench \
    --concept 'tench' \
    --prompts_csv 'data/small_imagenet_prompts.csv' \
    --output_log 'DoubleResult_Tench.log'

echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="

# change num_variations to 1