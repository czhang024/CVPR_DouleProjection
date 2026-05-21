import os
import csv
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
from datetime import datetime
import random
from collections import defaultdict
from safetensors.torch import load_file

def set_deterministic_mode(seed=42):
    """Set seeds for all random number generators to ensure reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Set deterministic mode with seed: {seed}")

class ErasedModelGenerationPipeline:
    def __init__(self, erased_model_path, model_id="CompVis/stable-diffusion-v1-4", 
                 device="cuda", output_dir="generated_images_erased", resolution=512,
                 batch_size=1):
        """
        Initialize image generation pipeline with erased model
        
        Args:
            erased_model_path: Path to the erased model
            model_id: Base Stable Diffusion model ID
            device: Device to run on (cuda/cpu)
            output_dir: Output directory
            resolution: Image resolution
            batch_size: Batch size (number of images to generate simultaneously)
        """
        self.device = device
        self.output_dir = output_dir
        self.resolution = resolution
        self.erased_model_path = erased_model_path
        self.batch_size = batch_size
        
        # Clear and create output directory
        if os.path.exists(output_dir):
            import shutil
            print(f"Output directory exists: {output_dir}")
            print(f"Clearing existing files...")
            shutil.rmtree(output_dir)
            print(f"✓ Directory cleared")
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"✓ Output directory created: {output_dir}")
        
        # Load model
        print(f"Loading Stable Diffusion model: {model_id}")
        print(f"Target resolution: {resolution}x{resolution}")
        print(f"Batch size: {batch_size}")
        
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            # torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        ).to(self.device)
        
        # Load erased weights
        print(f"\nLoading erased model weights from: {erased_model_path}")
        edited_weights = load_file(erased_model_path)
        self.pipe.unet.load_state_dict(edited_weights, strict=False)
        print(f"✓ Loaded {len(edited_weights)} edited parameters")
        
        # Optimization settings
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("✓ Enabled xformers memory efficient attention")
        except:
            print("✗ xformers not available, using default attention")
        
        # Add VAE optimization
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        
    def load_prompts_from_csv(self, csv_path):
        """Load prompts and related information from CSV file"""
        prompts_data = []
        objects_prompts = defaultdict(list)
        
        print(f"\nLoading prompts from: {csv_path}")
        
        if not os.path.exists(csv_path):
            print(f"✗ CSV file not found: {csv_path}")
            return prompts_data, objects_prompts
            
        with open(csv_path, 'r', encoding='utf-8') as file:
            # Read first line to check format
            first_line = file.readline().strip()
            file.seek(0)  # Return to beginning of file
            
            # Check if first column has column name (whether first character is comma)
            if first_line.startswith(','):
                # First column has no name, manually specify column names
                reader = csv.DictReader(file, fieldnames=['index', 'case_number', 'prompt', 'evaluation_seed', 'class'])
                next(reader)  # Skip the original header line
            else:
                # Normal CSV file with complete column names
                reader = csv.DictReader(file)
            
            for row in reader:
                # Extract data
                try:
                    # Handle possible column name variations (class vs object vs artist)
                    object_class = row.get('class', row.get('object', row.get('artist', ''))).strip()
                    
                    prompt_info = {
                        'index': int(row.get('index', row.get('', 0))),  # Compatible with both cases
                        'case_number': int(row.get('case_number', 0)),
                        'prompt': row.get('prompt', '').strip().strip('"'),  # Remove possible quotes
                        'evaluation_seed': int(row.get('evaluation_seed', 0)) if row.get('evaluation_seed') else None,
                        'object': object_class  # Uniformly use object as key name
                    }
                    
                    if prompt_info['prompt']:  # Ensure prompt is not empty
                        prompts_data.append(prompt_info)
                        objects_prompts[object_class].append(prompt_info)
                        
                except Exception as e:
                    print(f"Warning: Error parsing row: {row}, Error: {e}")
                    continue
        
        print(f"✓ Loaded {len(prompts_data)} prompts")
        print(f"✓ Found {len(objects_prompts)} unique objects:")
        
        # Display object statistics
        for obj, prompt_list in sorted(objects_prompts.items()):
            print(f"  - {obj}: {len(prompt_list)} prompts")
        
        return prompts_data, objects_prompts
    
    def generate_images_from_prompts(self, prompts_data, objects_prompts, 
                                   num_variations=1, base_seed_offset=0, global_seed=42):
        """
        Generate images based on prompts from CSV (supports batch processing)
        
        Args:
            prompts_data: List of all prompt data
            objects_prompts: Prompts grouped by object
            num_variations: Number of variations per prompt
            base_seed_offset: Base seed offset
            global_seed: Global seed (used when CSV has no evaluation_seed)
        """
        total_images = len(prompts_data) * num_variations
        print(f"\nTotal images to generate: {total_images}")
        
        # if self.batch_size > 1:
        #     print(f"✓ Batch processing enabled: generating {self.batch_size} images simultaneously")
        #     print(f"Estimated time on RTX 3090: ~{total_images * 2 / 60 / self.batch_size:.1f} minutes")
        # else:
        #     print(f"Estimated time on RTX 3090: ~{total_images * 2 / 60:.1f} minutes")
        
        # Check if evaluation_seed exists
        has_eval_seed = prompts_data[0]['evaluation_seed'] is not None if prompts_data else False
        
        if has_eval_seed:
            print("✓ Using evaluation_seed from CSV")
        else:
            print(f"✓ CSV does not contain evaluation_seed, using global_seed={global_seed}")
        
        # Create progress bar
        pbar = tqdm(total=total_images, desc="Generating images")
        
        # Record generation information
        generation_log = []
        
        # Don't create subdirectories, all images in main output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save all prompts to a file
        prompts_file = os.path.join(self.output_dir, "all_prompts.txt")
        with open(prompts_file, 'w', encoding='utf-8') as f:
            f.write("Index\tObject\tCase\tPrompt\n")
            f.write("="*80 + "\n")
            for prompt_info in prompts_data:
                f.write(f"{prompt_info['index']}\t{prompt_info['object']}\t"
                       f"{prompt_info['case_number']}\t{prompt_info['prompt']}\n")
        
        # Prepare all tasks
        tasks = []
        for prompt_info in prompts_data:
            for var_idx in range(num_variations):
                tasks.append((prompt_info, var_idx))
        
        # Batch processing generation
        for batch_start in range(0, len(tasks), self.batch_size):
            batch_tasks = tasks[batch_start:batch_start + self.batch_size]
            batch_prompts = []
            batch_seeds = []
            batch_metadata = []
            
            for prompt_info, var_idx in batch_tasks:
                index = prompt_info['index']
                prompt = prompt_info['prompt']
                case_number = prompt_info['case_number']
                eval_seed = prompt_info['evaluation_seed']
                object_name = prompt_info['object']
                
                # Decide which seed to use
                if eval_seed is not None:
                    seed = eval_seed + var_idx + base_seed_offset
                else:
                    seed = global_seed + index + var_idx + base_seed_offset
                
                batch_prompts.append(prompt)
                batch_seeds.append(seed)
                batch_metadata.append({
                    'index': index,
                    'object': object_name,
                    'case_number': case_number,
                    'eval_seed': eval_seed,
                    'var_idx': var_idx,
                    'seed': seed
                })
            
            # Update progress bar description
            first_meta = batch_metadata[0]
            if len(batch_metadata) > 1:
                pbar.set_description(f"Generating batch: {first_meta['object']} + {len(batch_metadata)-1} more")
            else:
                pbar.set_description(f"Generating: {first_meta['object']} - case {first_meta['case_number']}")
            
            try:
                # Batch generate images
                with torch.no_grad():
                    if self.batch_size == 1 or len(batch_prompts) == 1:
                        # Single image generation
                        generator = torch.Generator(device=self.device).manual_seed(batch_seeds[0])
                        images = self.pipe(
                            batch_prompts[0],
                            generator=generator,
                            num_inference_steps=50,
                            guidance_scale=7.5,
                            height=self.resolution,
                            width=self.resolution
                        ).images
                    else:
                        # Batch generation - uses same seed for entire batch
                        # Note: SD's batch generation uses same noise initialization for all prompts
                        # If completely different results are needed, use batch_size=1
                        generator = torch.Generator(device=self.device).manual_seed(batch_seeds[0])
                        images = self.pipe(
                            batch_prompts,
                            generator=generator,
                            num_inference_steps=50,
                            guidance_scale=7.5,
                            height=self.resolution,
                            width=self.resolution,
                            num_images_per_prompt=1
                        ).images
                
                # Save images
                for idx, (image, meta) in enumerate(zip(images, batch_metadata)):
                    object_name_safe = meta['object'].replace(' ', '_').replace("'", "")
                    
                    # Generate filename
                    if num_variations > 1:
                        filename = f"{meta['index']:03d}_{object_name_safe}_case{meta['case_number']:03d}_var{meta['var_idx']:02d}.png"
                    else:
                        filename = f"{meta['index']:03d}_{object_name_safe}_case{meta['case_number']:03d}.png"
                    
                    filepath = os.path.join(self.output_dir, filename)
                    image.save(filepath, optimize=True)
                    
                    # Record generation information
                    log_entry = {
                        'index': meta['index'],
                        'object': meta['object'],
                        'object_safe': object_name_safe,
                        'case_number': meta['case_number'],
                        'prompt': batch_prompts[idx],
                        'evaluation_seed': meta['eval_seed'] if meta['eval_seed'] is not None else 'N/A',
                        'generation_seed': meta['seed'],
                        'variation': meta['var_idx'],
                        'filename': filename,
                        'filepath': filepath,
                        'resolution': f"{self.resolution}x{self.resolution}"
                    }
                    generation_log.append(log_entry)
                    pbar.update(1)
                    
            except Exception as e:
                print(f"\nError generating batch starting at index {batch_metadata[0]['index']}: {str(e)}")
                print(f"Batch size: {len(batch_prompts)}")
                # Skip failed batch
                pbar.update(len(batch_prompts))
            
            # Periodically clear GPU cache
            if (batch_start // self.batch_size) % 5 == 0:
                torch.cuda.empty_cache()
        
        pbar.close()
        
        # Save generation log
        self.save_generation_log(generation_log)
        
        return generation_log
    
    def save_generation_log(self, generation_log):
        """Save generation log to CSV file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = os.path.join(self.output_dir, f"generation_log_{timestamp}.csv")
        
        with open(log_path, 'w', newline='', encoding='utf-8') as file:
            fieldnames = ['index', 'object', 'object_safe', 'case_number', 'prompt', 
                         'evaluation_seed', 'generation_seed', 'variation',
                         'filename', 'filepath', 'resolution']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(generation_log)
        
        print(f"\nGeneration log saved to: {log_path}")
        
        # Generate statistics report
        self.generate_statistics_report(generation_log, timestamp)
    
    def generate_statistics_report(self, generation_log, timestamp):
        """Generate statistics report"""
        report_path = os.path.join(self.output_dir, f"statistics_{timestamp}.txt")
        
        # Statistics information
        objects_count = defaultdict(int)
        has_eval_seed = False
        
        for entry in generation_log:
            objects_count[entry['object']] += 1
            if entry['evaluation_seed'] != 'N/A':
                has_eval_seed = True
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("GENERATION STATISTICS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Model: {self.erased_model_path}\n")
            f.write(f"Batch Size: {self.batch_size}\n")
            f.write(f"Generation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            if has_eval_seed:
                f.write(f"Seed Mode: Using evaluation_seed from CSV\n\n")
            else:
                f.write(f"Seed Mode: Using global_seed + index (CSV has no evaluation_seed)\n\n")
            
            f.write(f"Total Images Generated: {len(generation_log)}\n")
            f.write(f"Resolution: {generation_log[0]['resolution'] if generation_log else 'N/A'}\n")
            f.write(f"Output Directory: {self.output_dir}\n\n")
            
            f.write("Images per Object:\n")
            f.write("-" * 40 + "\n")
            for obj, count in sorted(objects_count.items()):
                f.write(f"  {obj}: {count} images\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("File Naming Convention:\n")
            f.write("-" * 30 + "\n")
            f.write("Format: {index}_{object_name}_case{case_number}.png\n")
            f.write("- index: CSV row index (3 digits)\n")
            f.write("- object_name: Object name with spaces replaced by underscores\n")
            f.write("- case_number: Case number from CSV (3 digits)\n")
            f.write("\nExample: 030_gas_pump_case030.png\n")
            f.write("\nAll images are saved directly in the output directory\n")
            f.write("=" * 60 + "\n")
        
        print(f"Statistics report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate images using erased model')
    parser.add_argument('--erased_model', type=str, 
                       default='esd_erase_test_results/esd_golf_ball_strength0.8_esd-x.safetensors',
                       help='Path to the erased model')
    parser.add_argument('--prompt_csv', type=str, default='data/small_imagenet_prompts.csv', 
                       help='Path to the CSV file containing prompts')
    parser.add_argument('--output_dir', type=str, default='generated_erased_golf_ball', 
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', 
                       choices=['cuda', 'cpu'], help='Device to run on')
    parser.add_argument('--model_id', type=str, default='CompVis/stable-diffusion-v1-4', 
                       help='Base Stable Diffusion model ID')
    parser.add_argument('--global_seed', type=int, default=42, 
                       help='Global seed for reproducibility (used when CSV has no evaluation_seed)')
    parser.add_argument('--resolution', type=int, default=512, 
                       help='Image resolution (square)')
    parser.add_argument('--num_variations', type=int, default=1,
                       help='Number of variations per prompt')
    parser.add_argument('--base_seed_offset', type=int, default=0, 
                       help='Base seed offset for generating different batches')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for parallel generation (higher values are faster but use more memory)')
    
    args = parser.parse_args()
    
    # Set deterministic mode
    set_deterministic_mode(args.global_seed)
    
    # Check CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    else:
        print(f"Using device: {args.device}")
        if args.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Initialize generation pipeline (using erased model)
    pipeline = ErasedModelGenerationPipeline(
        erased_model_path=args.erased_model,
        model_id=args.model_id,
        device=args.device,
        output_dir=args.output_dir,
        resolution=args.resolution,
        batch_size=args.batch_size
    )
    
    # Load prompts
    prompts_data, objects_prompts = pipeline.load_prompts_from_csv(args.prompt_csv)
    
    if not prompts_data:
        print("No prompts found in the CSV file!")
        return
    
    # Generate images
    print(f"\n{'='*60}")
    print(f"Starting image generation with erased model")
    print(f"Batch size: {args.batch_size}")
    print(f"Total prompts: {len(prompts_data)}")
    print(f"Resolution: {args.resolution}x{args.resolution}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}\n")
    
    generation_log = pipeline.generate_images_from_prompts(
        prompts_data, 
        objects_prompts,
        num_variations=args.num_variations,
        base_seed_offset=args.base_seed_offset,
        global_seed=args.global_seed
    )
    
    print(f"\n✓ Generation complete! Total images generated: {len(generation_log)}")
    print(f"Images saved to: {args.output_dir}")

if __name__ == "__main__":
    main()