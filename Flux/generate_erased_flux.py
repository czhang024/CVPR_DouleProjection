"""
generate_erased_flux.py
Generate images using a FLUX.1-schnell model whose transformer weights have been
partially replaced by the Double Proxy (or UCE) erasure method.

The erased model is stored as a .safetensors file containing only the modified
layer weights (e.g. context_embedder.weight, text_embedder.linear_1.weight).
These are loaded with strict=False so only the relevant keys are overwritten.
"""

import os
import csv
import random
import argparse
import shutil
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
from diffusers import FluxPipeline
from safetensors.torch import load_file
from PIL import Image
from tqdm import tqdm


def set_deterministic_mode(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Set deterministic mode with seed: {seed}")


class ErasedFluxGenerationPipeline:
    def __init__(self,
                 erased_model_path,
                 model_id='black-forest-labs/FLUX.1-schnell',
                 device='cuda',
                 output_dir='generated_erased_flux',
                 resolution=512,
                 batch_size=1,
                 num_inference_steps=4):
        self.device = device
        self.output_dir = output_dir
        self.resolution = resolution
        self.erased_model_path = erased_model_path
        self.batch_size = batch_size
        self.num_inference_steps = num_inference_steps

        # Clear output directory if it exists
        if os.path.exists(output_dir):
            print(f"Clearing existing output directory: {output_dir}")
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        print(f"✓ Output directory: {output_dir}")

        print(f"Loading FLUX model: {model_id}")
        self.pipe = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        ).to(device)

        # Load erased transformer weights
        print(f"Loading erased weights from: {erased_model_path}")
        edited_weights = load_file(erased_model_path)

        # The safetensors keys look like "context_embedder.weight"
        # We need to map them into the transformer's state dict
        transformer_state = self.pipe.transformer.state_dict()
        matched = 0
        for key, tensor in edited_weights.items():
            if key in transformer_state:
                transformer_state[key] = tensor.to(torch.bfloat16)
                matched += 1
            else:
                print(f"  Warning: key '{key}' not found in transformer state dict, skipping")
        self.pipe.transformer.load_state_dict(transformer_state, strict=True)
        print(f"✓ Applied {matched} edited parameter tensors")

        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        # NOTE: do NOT call enable_xformers_memory_efficient_attention() for FLUX —
        # xformers ignores image_rotary_emb which breaks FLUX attention.

    # ── Prompt loading ───────────────────────────────────────────────────────
    def load_prompts_from_csv(self, csv_path):
        prompts_data = []
        objects_prompts = defaultdict(list)

        print(f"\nLoading prompts from: {csv_path}")
        if not os.path.exists(csv_path):
            print(f"✗ CSV file not found: {csv_path}")
            return prompts_data, objects_prompts

        with open(csv_path, 'r', encoding='utf-8') as fh:
            first_line = fh.readline().strip()
            fh.seek(0)
            if first_line.startswith(','):
                reader = csv.DictReader(
                    fh,
                    fieldnames=['index', 'case_number', 'prompt', 'evaluation_seed', 'class'])
                next(reader)
            else:
                reader = csv.DictReader(fh)

            for row in reader:
                try:
                    object_class = row.get('class', row.get('object', row.get('artist', ''))).strip()
                    eval_seed_raw = row.get('evaluation_seed')
                    prompt_info = {
                        'index': int(row.get('index', row.get('', 0))),
                        'case_number': int(row.get('case_number', 0)),
                        'prompt': row.get('prompt', '').strip().strip('"'),
                        'evaluation_seed': int(eval_seed_raw) if eval_seed_raw else None,
                        'object': object_class,
                    }
                    if prompt_info['prompt']:
                        prompts_data.append(prompt_info)
                        objects_prompts[object_class].append(prompt_info)
                except Exception as exc:
                    print(f"Warning: skipping row {row}: {exc}")

        print(f"✓ Loaded {len(prompts_data)} prompts across {len(objects_prompts)} objects:")
        for obj, pl in sorted(objects_prompts.items()):
            print(f"  - {obj}: {len(pl)} prompts")
        return prompts_data, objects_prompts

    # ── Image generation ─────────────────────────────────────────────────────
    def generate_images_from_prompts(self, prompts_data, objects_prompts,
                                     num_variations=1, base_seed_offset=0, global_seed=42):
        total_images = len(prompts_data) * num_variations
        print(f"\nTotal images to generate: {total_images}")

        has_eval_seed = bool(prompts_data and prompts_data[0]['evaluation_seed'] is not None)
        print("✓ Using evaluation_seed from CSV" if has_eval_seed
              else f"✓ Using global_seed={global_seed}")

        pbar = tqdm(total=total_images, desc="Generating images")
        generation_log = []
        os.makedirs(self.output_dir, exist_ok=True)

        self.log_dir = os.path.join(self.output_dir + '_logs')
        os.makedirs(self.log_dir, exist_ok=True)

        with open(os.path.join(self.log_dir, 'all_prompts.txt'), 'w', encoding='utf-8') as fh:
            fh.write("Index\tObject\tCase\tPrompt\n" + "=" * 80 + "\n")
            for pi in prompts_data:
                fh.write(f"{pi['index']}\t{pi['object']}\t{pi['case_number']}\t{pi['prompt']}\n")

        tasks = [(pi, vi) for pi in prompts_data for vi in range(num_variations)]

        for batch_start in range(0, len(tasks), self.batch_size):
            batch = tasks[batch_start: batch_start + self.batch_size]
            batch_prompts, batch_seeds, batch_meta = [], [], []

            for pi, var_idx in batch:
                seed = (pi['evaluation_seed'] + var_idx + base_seed_offset
                        if pi['evaluation_seed'] is not None
                        else global_seed + pi['index'] + var_idx + base_seed_offset)
                batch_prompts.append(pi['prompt'])
                batch_seeds.append(seed)
                batch_meta.append({
                    'index': pi['index'],
                    'object': pi['object'],
                    'case_number': pi['case_number'],
                    'eval_seed': pi['evaluation_seed'],
                    'var_idx': var_idx,
                    'seed': seed,
                    'prompt': pi['prompt'],
                })

            first = batch_meta[0]
            pbar.set_description(
                f"Generating: {first['object']} case {first['case_number']}"
                + (f" +{len(batch_meta)-1} more" if len(batch_meta) > 1 else ""))

            try:
                with torch.no_grad():
                    if self.batch_size == 1 or len(batch_prompts) == 1:
                        generator = torch.Generator(device=self.device).manual_seed(batch_seeds[0])
                        images = self.pipe(
                            batch_prompts[0],
                            generator=generator,
                            num_inference_steps=self.num_inference_steps,
                            guidance_scale=0.0,
                            height=self.resolution,
                            width=self.resolution,
                            max_sequence_length=256,
                        ).images
                    else:
                        generator = torch.Generator(device=self.device).manual_seed(batch_seeds[0])
                        images = self.pipe(
                            batch_prompts,
                            generator=generator,
                            num_inference_steps=self.num_inference_steps,
                            guidance_scale=0.0,
                            height=self.resolution,
                            width=self.resolution,
                            max_sequence_length=256,
                            num_images_per_prompt=1,
                        ).images

                for image, meta in zip(images, batch_meta):
                    obj_safe = meta['object'].replace(' ', '_').replace("'", "")
                    filename = (
                        f"{meta['index']:03d}_{obj_safe}_case{meta['case_number']:03d}_var{meta['var_idx']:02d}.png"
                        if num_variations > 1
                        else f"{meta['index']:03d}_{obj_safe}_case{meta['case_number']:03d}.png"
                    )
                    filepath = os.path.join(self.output_dir, filename)
                    image.save(filepath, optimize=True)
                    generation_log.append({
                        'index': meta['index'],
                        'object': meta['object'],
                        'object_safe': obj_safe,
                        'case_number': meta['case_number'],
                        'prompt': meta['prompt'],
                        'evaluation_seed': meta['eval_seed'] if meta['eval_seed'] is not None else 'N/A',
                        'generation_seed': meta['seed'],
                        'variation': meta['var_idx'],
                        'filename': filename,
                        'filepath': filepath,
                        'resolution': f"{self.resolution}x{self.resolution}",
                    })
                    pbar.update(1)

            except Exception as exc:
                print(f"\nError at index {batch_meta[0]['index']}: {exc}")
                pbar.update(len(batch_prompts))

            if (batch_start // self.batch_size) % 5 == 0:
                torch.cuda.empty_cache()

        pbar.close()
        self._save_log(generation_log)
        return generation_log

    def _save_log(self, generation_log):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = os.path.join(self.log_dir, f"generation_log_{ts}.csv")
        fieldnames = ['index', 'object', 'object_safe', 'case_number', 'prompt',
                      'evaluation_seed', 'generation_seed', 'variation',
                      'filename', 'filepath', 'resolution']
        with open(log_path, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(generation_log)
        print(f"\nGeneration log saved to: {log_path}")

        report_path = os.path.join(self.log_dir, f"statistics_{ts}.txt")
        obj_count = defaultdict(int)
        for e in generation_log:
            obj_count[e['object']] += 1
        with open(report_path, 'w', encoding='utf-8') as fh:
            fh.write("=" * 60 + "\n")
            fh.write("GENERATION STATISTICS REPORT\n")
            fh.write("=" * 60 + "\n\n")
            fh.write(f"Erased Model: {self.erased_model_path}\n")
            fh.write(f"Batch Size: {self.batch_size}\n")
            fh.write(f"Steps: {self.num_inference_steps}\n")
            fh.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            fh.write(f"Total Images: {len(generation_log)}\n")
            fh.write(f"Resolution: {generation_log[0]['resolution'] if generation_log else 'N/A'}\n")
            fh.write(f"Output Directory: {self.output_dir}\n\n")
            fh.write("Images per Object:\n" + "-" * 40 + "\n")
            for obj, cnt in sorted(obj_count.items()):
                fh.write(f"  {obj}: {cnt} images\n")
        print(f"Statistics report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate images with erased FLUX.1-schnell model')
    parser.add_argument('--erased_model', type=str, required=True,
                        help='Path to the erased model .safetensors file')
    parser.add_argument('--prompt_csv', type=str, default='../SD/data/small_imagenet_prompts.csv')
    parser.add_argument('--output_dir', type=str, default='generated_erased_flux')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--model_id', type=str, default='black-forest-labs/FLUX.1-schnell')
    parser.add_argument('--global_seed', type=int, default=42)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--num_variations', type=int, default=10)
    parser.add_argument('--base_seed_offset', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_inference_steps', type=int, default=4,
                        help='Denoising steps (4 recommended for FLUX.1-schnell)')
    args = parser.parse_args()

    set_deterministic_mode(args.global_seed)

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    else:
        if args.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    pipeline = ErasedFluxGenerationPipeline(
        erased_model_path=args.erased_model,
        model_id=args.model_id,
        device=args.device,
        output_dir=args.output_dir,
        resolution=args.resolution,
        batch_size=args.batch_size,
        num_inference_steps=args.num_inference_steps,
    )

    prompts_data, objects_prompts = pipeline.load_prompts_from_csv(args.prompt_csv)
    if not prompts_data:
        print("No prompts found!")
        return

    print(f"\n{'='*60}")
    print(f"Erased FLUX generation")
    print(f"Total prompts: {len(prompts_data)}  |  Variations: {args.num_variations}")
    print(f"Total images : {len(prompts_data) * args.num_variations}")
    print(f"{'='*60}\n")

    generation_log = pipeline.generate_images_from_prompts(
        prompts_data,
        objects_prompts,
        num_variations=args.num_variations,
        base_seed_offset=args.base_seed_offset,
        global_seed=args.global_seed,
    )
    print(f"\n✓ Done. {len(generation_log)} images saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
