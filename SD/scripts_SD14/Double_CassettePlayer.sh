#!/bin/bash

set -e  # Exit on error

echo "=========================================="
echo "Step 1: Training erased model with Double Proxy"
echo "=========================================="


# Double Proxy UCE for erasing cassette player concept while preserving other ImageNette concepts
CUDA_VISIBLE_DEVICES=0 python Double_proxy.py \
    --edit_concepts "cassette player" \
    --guide_concepts "box" \
    --preserve_concepts "chain saw;church;gas pump;tench;garbage truck;english springer;golf ball;parachute;french horn" \
    --concept_type "object" \
    --save_dir "./models" \
    --exp_name "UCE_double_proxy_CassettePlayer"

echo "=========================================="
echo "Step 2: Generating images with erased model"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0 python generate_erased_small_imagenet.py \
    --erased_model models/UCE_double_proxy_CassettePlayer/diffusion_pytorch_model.safetensors \
    --output_dir generated_Double_CassettePlayer \
    --num_variations 10

echo "=========================================="
echo "Step 3: Comparing accuracy"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0 python compare_accuracy.py \
    --original_path generated_original_sd14 \
    --erased_path generated_Double_CassettePlayer \
    --concept 'cassette player' \
    --prompts_csv 'data/small_imagenet_prompts.csv' \
    --output_log 'DoubleResult_CassettePlayer.log'

echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
