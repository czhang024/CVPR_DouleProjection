#!/usr/bin/env python3
"""
compare_erasure_accuracy_fixed.py
Evaluation script that fixes the cassette player/tape player confusion issue
"""

# Copy all imports and class definitions from original compare_erasure_accuracy.py
import os
import argparse
import pandas as pd
import torch
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from tabulate import tabulate

# Define semantic equivalent class mappings
SEMANTIC_EQUIVALENT_CLASSES = {
    'cassette player': ['cassette player', 'tape player', 'radio'],  # These are semantically equivalent
    # Can add other equivalent mappings
}

class ErasureAccuracyEvaluator:
    def __init__(self, device='cuda:0', batch_size=250, topk=5):
        self.device = device
        self.batch_size = batch_size
        self.topk = topk
        
        # Load ResNet50 model
        self.weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=self.weights)
        self.model.to(device)
        self.model.eval()
        self.preprocess = self.weights.transforms()
        
    def classify_images(self, folder_path, prompts_df):
        """Classify images in the folder"""
        print(f"\nClassifying images in: {folder_path}")
        
        # Get all image files
        names = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if not names:
            raise ValueError(f"No images found in {folder_path}")
        
        print(f"Found {len(names)} images")
        
        # Track skipped files
        skipped_files = []
        
        # Prepare to store results
        results = {
            'filename': [],
            'case_number': []
        }
        for k in range(1, self.topk + 1):
            results[f'category_top{k}'] = []
            results[f'index_top{k}'] = []
            results[f'scores_top{k}'] = []
        
        # Batch process images
        for i in tqdm(range(0, len(names), self.batch_size), desc="Processing batches"):
            batch_names = names[i:i + self.batch_size]
            batch_images = []
            valid_names = []  # Track which images loaded successfully
            
            # Load images
            for name in batch_names:
                img_path = os.path.join(folder_path, name)
                try:
                    img = Image.open(img_path).convert('RGB')
                    batch_images.append(self.preprocess(img))
                    valid_names.append(name)
                except (OSError, IOError) as e:
                    print(f"\nWarning: Skipping corrupted image {name}: {e}")
                    skipped_files.append(name)
                    continue
            
            # Skip this batch if no valid images
            if len(batch_images) == 0:
                continue
            
            # Batch prediction
            batch_tensor = torch.stack(batch_images).to(self.device)
            with torch.no_grad():
                predictions = self.model(batch_tensor).softmax(1)
            
            probs, class_ids = torch.topk(predictions, self.topk, dim=1)
            
            # Save results - use valid_names instead of batch_names
            for j, name in enumerate(valid_names):
                results['filename'].append(name)
                # Extract case_number from filename
                try:
                    parts = name.split('_')
                    for part in parts:
                        if 'case' in part:
                            case_num = int(part.replace('case', '').replace('.png', '').replace('.jpg', ''))
                            results['case_number'].append(case_num)
                            break
                    else:
                        case_num = int(parts[0])
                        results['case_number'].append(case_num)
                except:
                    results['case_number'].append(-1)
                
                for k in range(1, self.topk + 1):
                    results[f'category_top{k}'].append(
                        self.weights.meta["categories"][class_ids[j, k-1].item()]
                    )
                    results[f'index_top{k}'].append(class_ids[j, k-1].item())
                    results[f'scores_top{k}'].append(probs[j, k-1].item())
        
        # Print summary of skipped files
        if skipped_files:
            print(f"\n⚠️  Warning: Skipped {len(skipped_files)} corrupted images:")
            for fname in skipped_files[:10]:  # Show first 10
                print(f"  - {fname}")
            if len(skipped_files) > 10:
                print(f"  ... and {len(skipped_files) - 10} more")
        
        # Create DataFrame and merge with prompts
        results_df = pd.DataFrame(results)
        
        # Merge prompts information
        prompts_df['case_number'] = prompts_df['case_number'].astype(int)
        results_df['case_number'] = results_df['case_number'].astype(int)
        
        merged_df = pd.merge(prompts_df, results_df, on='case_number', how='inner')
        
        return merged_df
    
    def compute_accuracy_by_class(self, df):
        """Compute accuracy for each class (considering semantic equivalent classes)"""
        # Create a new column to mark whether correct (considering equivalent classes)
        df['correct'] = False
        df['correct_semantic'] = False
        
        for _, row in df.iterrows():
            true_class = row['class'].strip().lower()
            pred_class = row['category_top1'].strip().lower()
            
            # Exact match
            if true_class == pred_class:
                df.loc[row.name, 'correct'] = True
                df.loc[row.name, 'correct_semantic'] = True
            
            # Semantic equivalence match
            elif true_class in SEMANTIC_EQUIVALENT_CLASSES:
                equivalent_classes = [c.lower() for c in SEMANTIC_EQUIVALENT_CLASSES[true_class]]
                if pred_class in equivalent_classes:
                    df.loc[row.name, 'correct_semantic'] = True
        
        # Compute accuracy for each class
        acc_table = df.groupby('class').agg({
            'correct': ['mean', 'sum', 'count'],
            'correct_semantic': ['mean', 'sum']
        }).reset_index()
        
        acc_table.columns = ['class', 'accuracy_strict', 'correct_count_strict', 
                           'total_count', 'accuracy_semantic', 'correct_count_semantic']
        
        return acc_table
    
    def compare_accuracy(self, original_df, erased_df, target_concept=None):
        """Compare accuracy between original and erased models"""
        # Compute accuracy
        acc_original = self.compute_accuracy_by_class(original_df)
        acc_erased = self.compute_accuracy_by_class(erased_df)
        
        # Merge results
        merged = pd.merge(
            acc_original, 
            acc_erased, 
            on='class', 
            suffixes=('_original', '_erased')
        )
        
        # Calculate accuracy change (using semantic accuracy)
        merged['accuracy_drop (%)'] = (merged['accuracy_semantic_original'] - 
                                      merged['accuracy_semantic_erased']) * 100
        merged['accuracy_drop_strict (%)'] = (merged['accuracy_strict_original'] - 
                                             merged['accuracy_strict_erased']) * 100
        
        # Reorder columns, primarily showing semantic accuracy
        merged = merged[[
            'class', 
            'accuracy_semantic_original',
            'accuracy_semantic_erased',
            'accuracy_drop (%)',
            'accuracy_strict_original',
            'accuracy_strict_erased', 
            'accuracy_drop_strict (%)',
            'correct_count_semantic_original',
            'total_count_original',
            'correct_count_semantic_erased',
            'total_count_erased'
        ]]
        
        # If target concept is specified, put it at the top
        if target_concept and target_concept in merged['class'].values:
            target_row = merged[merged['class'] == target_concept]
            other_rows = merged[merged['class'] != target_concept]
            merged = pd.concat([target_row, other_rows], ignore_index=True)
        
        return merged

def print_terminal_summary(comparison_df, target_concept):
    """Print concise result summary in terminal"""
    # Print statistical summary only
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    if target_concept in comparison_df['class'].values:
        target_stats = comparison_df[comparison_df['class'] == target_concept].iloc[0]
        print(f"\nTarget Concept ({target_concept}):")
        print(f"  • Original Accuracy (semantic): {target_stats['accuracy_semantic_original']:.1%}")
        print(f"  • Erased Accuracy (semantic):   {target_stats['accuracy_semantic_erased']:.1%}")
        print(f"  • Accuracy Drop:                {target_stats['accuracy_drop (%)']:.1f}%")
        print(f"  • Original Accuracy (strict):   {target_stats['accuracy_strict_original']:.1%}")
        print(f"  • Erased Accuracy (strict):     {target_stats['accuracy_strict_erased']:.1%}")
        
        other_classes = comparison_df[comparison_df['class'] != target_concept]
        if not other_classes.empty:
            print(f"\nOther Classes (Average):")
            print(f"  • Original Accuracy: {other_classes['accuracy_semantic_original'].mean():.1%}")
            print(f"  • Erased Accuracy:   {other_classes['accuracy_semantic_erased'].mean():.1%}")
            print(f"  • Average Drop:      {other_classes['accuracy_drop (%)'].mean():.1f}%")
    
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(
        description="Compare classification accuracy before and after concept erasure"
    )
    parser.add_argument('--original_path', type=str, required=True,
                       help="Path to original model generated images")
    parser.add_argument('--erased_path', type=str, required=True,
                       help="Path to erased model generated images")
    parser.add_argument('--prompts_csv', type=str, required=True,
                       help="Path to small_imagenet_prompts.csv")
    parser.add_argument('--concept', type=str, required=True,
                       help="Target concept that was erased")
    parser.add_argument('--output_log', type=str, default=None,
                       help="Path to save results log")
    parser.add_argument('--device', type=str, default='cuda:0',
                       help="Device to use for classification")
    parser.add_argument('--batch_size', type=int, default=250,
                       help="Batch size for classification")
    
    args = parser.parse_args()
    
    # Set output log path
    if args.output_log is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_log = f"erasure_accuracy_comparison_{args.concept}_{timestamp}.log"
    
    # Initialize evaluator
    print("Initializing evaluator...")
    evaluator = ErasureAccuracyEvaluator(
        device=args.device,
        batch_size=args.batch_size
    )
    
    # Load prompts
    print(f"\nLoading prompts from: {args.prompts_csv}")
    prompts_df = pd.read_csv(args.prompts_csv)
    print(f"Loaded {len(prompts_df)} prompts")
    
    # Classify original images
    print("\n" + "="*60)
    print("Step 1: Classifying original images")
    print("="*60)
    original_results = evaluator.classify_images(args.original_path, prompts_df)
    
    # Classify erased model images
    print("\n" + "="*60)
    print("Step 2: Classifying erased model images")
    print("="*60)
    erased_results = evaluator.classify_images(args.erased_path, prompts_df)
    
    # Compare accuracy
    print("\n" + "="*60)
    print("Step 3: Comparing accuracy")
    print("="*60)
    comparison = evaluator.compare_accuracy(
        original_results, 
        erased_results, 
        target_concept=args.concept
    )
    
    # Print concise results in terminal
    print_terminal_summary(comparison, args.concept)
    
    # # Save results
    # if args.output_log:
    #     comparison.to_csv(args.output_log.replace('.log', '_comparison.csv'), index=False)
    #     print(f"\nDetailed results saved to: {args.output_log.replace('.log', '_comparison.csv')}")

if __name__ == '__main__':
    main()