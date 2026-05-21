#!/bin/bash

set -e  # Exit on error

echo "=========================================="
echo "Step 1: Training erased model with Double Proxy (FLUX)"
echo "=========================================="

# NOTE: SD 1.4 tokenizer splits "chain saw" as ['chain', 'saw'] → replace_indices "1" erases "saw".
# FLUX T5 tokenizer also splits "chain saw" into ['chain', '▁saw'] → same index 1 applies.
CUDA_VISIBLE_DEVICES=0 python Double_proxy.py \
    --edit_concepts "chain saw" \
    --guide_concepts "stick" \
    --preserve_concepts "cassette player;church;gas pump;tench;garbage truck;english springer;golf ball;parachute;french horn" \
    --concept_type "object" \
    --save_dir "./models" \
    --exp_name "flux_double_proxy_ChainSaw"

echo "=========================================="
echo "Step 2: Generating images with erased FLUX model"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0 python generate_erased_flux.py \
    --erased_model models/flux_double_proxy_ChainSaw.safetensors \
    --output_dir generated_Double_ChainSaw_flux \
    --prompt_csv ../SD/data/small_imagenet_prompts.csv \
    --num_variations 10

echo "=========================================="
echo "Step 3: Comparing accuracy"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0 python compare_accuracy.py \
    --original_path generated_original_flux \
    --erased_path generated_Double_ChainSaw_flux \
    --concept 'chain saw' \
    --prompts_csv '../SD/data/small_imagenet_prompts.csv' \
    --output_log 'DoubleResult_ChainSaw_flux.log'

echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
