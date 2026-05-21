#!/bin/bash

set -e  # Exit on error

echo "=========================================="
echo "Step 1: Training erased model with Double Proxy (FLUX)"
echo "=========================================="

# "garbage truck" → tokens ['garbage', 'truck'] in both CLIP and T5.
CUDA_VISIBLE_DEVICES=0 python Double_proxy.py \
    --edit_concepts "garbage truck" \
    --guide_concepts "bus" \
    --preserve_concepts "cassette player;chain saw;church;gas pump;golf ball;tench;english springer;parachute;french horn" \
    --concept_type "object" \
    --save_dir "./models" \
    --exp_name "flux_double_proxy_GarbageTruck"

echo "=========================================="
echo "Step 2: Generating images with erased FLUX model"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0 python generate_erased_flux.py \
    --erased_model models/flux_double_proxy_GarbageTruck.safetensors \
    --output_dir generated_Double_GarbageTruck_flux \
    --prompt_csv ../SD/data/small_imagenet_prompts.csv \
    --num_variations 10

echo "=========================================="
echo "Step 3: Comparing accuracy"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0 python compare_accuracy.py \
    --original_path generated_original_flux \
    --erased_path generated_Double_GarbageTruck_flux \
    --concept 'garbage truck' \
    --prompts_csv '../SD/data/small_imagenet_prompts.csv' \
    --output_log 'DoubleResult_GarbageTruck_flux.log'

echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
