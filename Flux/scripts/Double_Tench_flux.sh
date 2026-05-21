#!/bin/bash

set -e  # Exit on error

echo "=========================================="
echo "Step 1: Training erased model with Double Proxy (FLUX)"
echo "=========================================="

# NOTE on tokenization: In SD 1.4 (CLIP tokenizer), "tench" splits into ['ten', 'ch</w>'] (2 tokens),
# so replace_indices cannot be "0" alone. In FLUX T5 tokenizer, "tench" is a SINGLE token.
# Therefore we use the default (all) which correctly erases the single T5 token.
CUDA_VISIBLE_DEVICES=0 python Double_proxy.py \
    --edit_concepts "tench" \
    --guide_concepts "cucumber" \
    --preserve_concepts "cassette player;chain saw;church;gas pump;golf ball;garbage truck;english springer;parachute;french horn" \
    --concept_type "object" \
    --save_dir "./models" \
    --exp_name "flux_double_proxy_Tench"

echo "=========================================="
echo "Step 2: Generating images with erased FLUX model"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0 python generate_erased_flux.py \
    --erased_model models/flux_double_proxy_Tench.safetensors \
    --output_dir generated_Double_Tench_flux \
    --prompt_csv ../SD/data/small_imagenet_prompts.csv \
    --num_variations 10

echo "=========================================="
echo "Step 3: Comparing accuracy"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0 python compare_accuracy.py \
    --original_path generated_original_flux \
    --erased_path generated_Double_Tench_flux \
    --concept 'tench' \
    --prompts_csv '../SD/data/small_imagenet_prompts.csv' \
    --output_log 'DoubleResult_Tench_flux.log'

echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
