#!/bin/bash


set -e  # Exit on error

echo "=========================================="
echo "Step 1: Training erased model with Double Proxy"
echo "=========================================="


# UCE Original for erasing cassette player concept while preserving other ImageNette concepts
CUDA_VISIBLE_DEVICES=1 python UCE_original.py \
    --edit_concepts "cassette player" \
    --guide_concepts "box" \
    --preserve_concepts "chain saw;church;gas pump;tench;garbage truck;english springer;golf ball;parachute;french horn" \
    --concept_type "object" \
    --lamb 0.5 \
    --preserve_scale 1.0 \
    --erase_scale 1.0 \
    --save_dir "./models" \
    --exp_name "UCE_original_CassettePlayer"

echo "=========================================="
echo "Step 2: Generating images with erased model"
echo "=========================================="

CUDA_VISIBLE_DEVICES=1 python generate_erased_small_imagenet.py \
    --erased_model models/UCE_original_CassettePlayer/diffusion_pytorch_model.safetensors \
    --output_dir generated_UCE_CassettePlayer \
    --num_variations 10

echo "=========================================="
echo "Step 3: Comparing accuracy"
echo "=========================================="

CUDA_VISIBLE_DEVICES=1 python compare_accuracy.py \
    --original_path generated_original_sd14 \
    --erased_path generated_UCE_CassettePlayer \
    --concept 'cassette player' \
    --prompts_csv 'data/small_imagenet_prompts.csv' \
    --output_log 'UCEResult_CassettePlayer.log'


echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
