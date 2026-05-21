#!/bin/bash

set -e  # Exit on error

echo "=========================================="
echo "Step 1: Training erased model with Double Proxy (FLUX)"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0 python Double_proxy.py \
    --edit_concepts "gas pump" \
    --guide_concepts "booth" \
    --preserve_concepts "cassette player;chain saw;church;english springer;french horn;tench;garbage truck;golf ball;parachute" \
    --concept_type "object" \
    --save_dir "./models" \
    --exp_name "flux_double_proxy_GasPump"

echo "=========================================="
echo "Step 2: Generating images with erased FLUX model"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0 python generate_erased_flux.py \
    --erased_model models/flux_double_proxy_GasPump.safetensors \
    --output_dir generated_Double_GasPump_flux \
    --prompt_csv ../SD/data/small_imagenet_prompts.csv \
    --num_variations 10

echo "=========================================="
echo "Step 3: Comparing accuracy"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0 python compare_accuracy.py \
    --original_path generated_original_flux \
    --erased_path generated_Double_GasPump_flux \
    --concept 'gas pump' \
    --prompts_csv '../SD/data/small_imagenet_prompts.csv' \
    --output_log 'DoubleResult_GasPump_flux.log'

echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
