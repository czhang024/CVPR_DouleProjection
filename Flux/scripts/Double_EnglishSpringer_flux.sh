#!/bin/bash

set -e  # Exit on error

echo "=========================================="
echo "Step 1: Training erased model with Double Proxy (FLUX)"
echo "=========================================="

# "english springer" → tokens ['english', 'springer'] in both CLIP and T5.
# No replace_indices: erase both tokens (default: all).
CUDA_VISIBLE_DEVICES=0 python Double_proxy.py \
    --edit_concepts "english springer" \
    --guide_concepts "cat" \
    --preserve_concepts "cassette player;chain saw;church;gas pump;tench;garbage truck;golf ball;parachute;french horn" \
    --concept_type "object" \
    --save_dir "./models" \
    --exp_name "flux_double_proxy_EnglishSpringer"

echo "=========================================="
echo "Step 2: Generating images with erased FLUX model"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0 python generate_erased_flux.py \
    --erased_model models/flux_double_proxy_EnglishSpringer.safetensors \
    --output_dir generated_Double_EnglishSpringer_flux \
    --prompt_csv ../SD/data/small_imagenet_prompts.csv \
    --num_variations 10

echo "=========================================="
echo "Step 3: Comparing accuracy"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0 python compare_accuracy.py \
    --original_path generated_original_flux \
    --erased_path generated_Double_EnglishSpringer_flux \
    --concept 'english springer' \
    --prompts_csv '../SD/data/small_imagenet_prompts.csv' \
    --output_log 'DoubleResult_EnglishSpringer_flux.log'

echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
