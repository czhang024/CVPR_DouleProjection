import torch
torch.set_grad_enabled(False)
import argparse
import os
import copy
import time

from safetensors.torch import save_file
from diffusers import DiffusionPipeline


def UCE_double_proxy(pipe, edit_concepts, guide_concepts, preserve_concepts, save_dir, exp_name, replace_indices=None):
    """
    Double Proxy Solution for Concept Erasing
    
    This implements the two-step optimization:
    Step 1: Compute proxy vector v_i* as projection onto span of guide + preserve concepts
    Step 2: Solve constrained optimization for W in the left null space of V
    
    Args:
        replace_indices: List of indices specifying which word(s) to replace in each erase concept.
                        None means replace all words (default behavior).
                        For example: [None, [1], [0, 2]] means:
                        - First concept: replace all words
                        - Second concept: replace only word at index 1 (0-indexed)
                        - Third concept: replace words at indices 0 and 2
    """
    start_time = time.time()
    
    # Prepare the cross attention weights required to do UCE
    uce_modules = []
    uce_module_names = []
    for name, module in pipe.unet.named_modules():
        if 'attn2' in name and (name.endswith('to_v') or name.endswith('to_k')):
            uce_modules.append(module)
            uce_module_names.append(name)
    original_modules = copy.deepcopy(uce_modules)
    uce_modules = copy.deepcopy(uce_modules)

    # Collect text embeddings for all concepts
    # Store as list of token embeddings for multi-word concepts
    uce_erase_embeds = {}
    all_concepts = list(set(edit_concepts + guide_concepts + preserve_concepts))  # Remove duplicates
    
    for e in all_concepts:
        if e in uce_erase_embeds or not e:  # Skip empty strings
            continue
        t_emb = pipe.encode_prompt(prompt=e,
                                   device=device,
                                   num_images_per_prompt=1,
                                   do_classifier_free_guidance=False)
    
        # Get attention mask to find valid tokens
        tokenizer_output = pipe.tokenizer(e,
                                          padding="max_length",
                                          max_length=pipe.tokenizer.model_max_length,
                                          truncation=True,
                                          return_tensors="pt")
        attention_mask = tokenizer_output['attention_mask']
        
        # Find valid token positions (excluding [CLS] at 0 and [SEP] at end)
        # attention_mask is 1 for valid tokens, 0 for padding
        valid_token_count = attention_mask.sum().item()
        
        # Extract embeddings for ALL valid tokens (excluding [CLS] and [SEP])
        # For "gas pump": tokens are [CLS, gas, pump, SEP, PAD, PAD, ...]
        # We store embeddings for tokens 1 and 2 (gas, pump) separately
        # Store as list: each element is [1, 1, n] for one token
        token_embeddings = []
        for token_idx in range(1, valid_token_count - 1):
            token_embeddings.append(t_emb[0][:, token_idx:token_idx+1, :])  # Shape: [1, 1, n]
        
        uce_erase_embeds[e] = token_embeddings
        print(f"  '{e}': {len(token_embeddings)} token(s)")
    
    print(f"\nEmbeddings collected for {len(uce_erase_embeds)} concepts")

    # Initialize replace_indices if not provided (default: replace all words)
    if replace_indices is None:
        replace_indices = [None] * len(edit_concepts)
    
    # Validate replace_indices length
    if len(replace_indices) != len(edit_concepts):
        raise ValueError(f"replace_indices length ({len(replace_indices)}) must match edit_concepts length ({len(edit_concepts)})")
    
    print(f"\nReplacement configuration:")
    for i, (concept, indices) in enumerate(zip(edit_concepts, replace_indices)):
        num_tokens = len(uce_erase_embeds[concept])
        if indices is None:
            print(f"  '{concept}': replacing ALL {num_tokens} word(s)")
        else:
            print(f"  '{concept}': replacing word(s) at index/indices {indices} (out of {num_tokens} total)")

    ###### Double Proxy Algorithm Implementation
    for module_idx, module in enumerate(original_modules):
        # Get original weight of the model (W_0)
        W_0 = module.weight  # Shape: [p, n]
        
        # Process each erase concept
        for erase_idx, erase_concept in enumerate(edit_concepts):
            # Get the token embeddings for the concept to erase
            # For multi-word concepts, this is a list of [1, 1, n] tensors
            erase_token_embeddings = uce_erase_embeds[erase_concept]
            
            # Determine which tokens to process based on replace_indices
            indices_to_replace = replace_indices[erase_idx]
            if indices_to_replace is None:
                # Replace all tokens
                tokens_to_process = list(range(len(erase_token_embeddings)))
            else:
                # Replace only specified tokens
                # Ensure indices are valid
                tokens_to_process = []
                for idx in indices_to_replace:
                    if 0 <= idx < len(erase_token_embeddings):
                        tokens_to_process.append(idx)
                    else:
                        print(f"  Warning: Index {idx} out of range for '{erase_concept}' (has {len(erase_token_embeddings)} tokens), skipping")
            
            # Process each selected token of the erase concept
            for token_idx in tokens_to_process:
                erase_token_emb = erase_token_embeddings[token_idx]
                v_i = erase_token_emb.squeeze(0).T  # Shape: [n, 1]
                
                # ===== STEP 1: Compute Proxy Vector v_i* =====
                # Collect ALL guide concept embeddings
                guide_embeddings = []
                
                # Add ALL tokens from ALL guide concepts
                for guide_concept in guide_concepts:
                    if guide_concept and guide_concept in uce_erase_embeds:
                        for guide_token_emb in uce_erase_embeds[guide_concept]:
                            guide_embeddings.append(guide_token_emb.squeeze(0).T)  # [n, 1]
                
                if len(guide_embeddings) == 0:
                    print(f"  Warning: No guide concepts available, skipping {erase_concept}")
                    continue
            
                # Build V matrix from guide embeddings for projection
                V_guide = torch.cat(guide_embeddings, dim=1)  # Shape: [n, num_guides]
                
                # For single guide: set v_i* = guide embedding directly (no scaling)
                # For multiple guides: use projection formula
                if len(guide_embeddings) == 1:
                    v_i_star = guide_embeddings[0]  # Shape: [n, 1]
                else:
                    # Compute v_i* using projection formula: v_i* = V(V^T V)^+ V^T v_i
                    VtV_guide = V_guide.T @ V_guide  # Shape: [num_guides, num_guides]
                    VtV_guide_pinv = torch.linalg.pinv(VtV_guide.float()).to(torch_dtype)
                    v_i_star = V_guide @ VtV_guide_pinv @ V_guide.T @ v_i  # Shape: [n, 1]
                
                            
                # ===== STEP 2: Constrained Optimization for W =====
                # Construct V_step2 matrix from ONLY preserve concepts (not guide concepts)
                V_step2_list = []
                
                # Add ALL tokens from all preserve concepts
                for preserve_concept in preserve_concepts:
                    if preserve_concept in uce_erase_embeds:
                        for preserve_token_emb in uce_erase_embeds[preserve_concept]:
                            V_step2_list.append(preserve_token_emb.squeeze(0).T)  # [n, 1]
                
                if len(V_step2_list) == 0:
                    print(f"  Warning: No preserve concepts, using unconstrained update")
                    # Unconstrained update: W = W_0 + (v_i_star - v_i) @ W_0^T / ||W_0^T||^2
                    # Simplified: just move v_i towards v_i_star
                    Delta_W = (W_0 @ v_i_star - W_0 @ v_i) @ v_i.T / (v_i.T @ v_i).item()
                    W_0 = W_0 + Delta_W
                    print(f"  Weight update applied (unconstrained) for {erase_concept} token {token_idx+1}")
                    continue
                
                # Stack into V_step2 matrix: [n, m_preserve]
                V_step2 = torch.cat(V_step2_list, dim=1)  # Shape: [n, m_preserve]
                
                # IMPORTANT: nn.Linear(V) computes V @ W^T, not W @ V
                # So the operation is: output = V @ W^T where V is [batch, seq, n] and W is [p, n]
                # To preserve outputs: V_step2 @ (W^T) = V_step2 @ (W_0^T)
                # This means: V_step2 @ (W - W_0)^T = 0
                # Which means: V_step2 @ (ΔW)^T = 0
                # Therefore: ΔW^T must be in the RIGHT null space of V_step2
                
                # SVD of V_step2^T to get its right null space
                # V_step2^T has shape [m_preserve, n], so after SVD: V_step2^T = U Σ Vh
                U, S, Vh = torch.linalg.svd(V_step2.T.float(), full_matrices=True)  # V_step2^T = U Σ Vh
                
                rank_V = torch.sum(S > 1e-6).item()
                
                # Right null space of V_step2^T: rows of Vh from rank_V onwards
                # These vectors satisfy V_step2^T @ u = 0
                # U_2 will have shape [n, n-r] where each column is a null space basis vector
                U_2 = Vh[rank_V:, :].T.to(torch_dtype)  # Shape: [n, n-r]
                
                    
                x = U_2.T @ v_i  # Shape: [n-r, 1]
                b = W_0 @ v_i_star - W_0 @ v_i  # Shape: [p, 1]
                
                # Closed-form solution: Z = b x^T / ||x||^2
                # Handle the case where x might be very small
                x_norm_sq = (x.T @ x).item()
                
                if x_norm_sq < 1e-10:
                    print(f"  Warning: ||x||^2 = {x_norm_sq} is very small, skipping this token")
                    continue
                
                Z = (b @ x.T) / x_norm_sq  # Shape: [p, n-r]
                
                # Compute ΔW = Z U_2^T
                Delta_W = Z @ U_2.T  # Shape: [p, n]
                
                # Update: W = W_0 + ΔW
                W_new = W_0 + Delta_W

                # Apply weight update
                W_0 = W_new
        
        # Set the updated weights
        uce_modules[module_idx].weight = torch.nn.Parameter(W_0)
    
    # Update the pipeline with the new weights
    for name, module in zip(uce_module_names, uce_modules):
        # Navigate through the nested structure to set the module
        parts = name.split('.')
        target = pipe.unet
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], module)
        
    # Save only the UNet model
    unet_save_path = os.path.join(save_dir, exp_name)
    os.makedirs(unet_save_path, exist_ok=True)
    pipe.unet.save_pretrained(unet_save_path)
    
    end_time = time.time()
    print(f'\n\nErased concepts using Double Proxy UCE\nModel edited in {end_time-start_time:.2f} seconds\n')
    print(f'UNet model saved to {unet_save_path}\n')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog = 'TrainUCE_DoubleProxy',
                    description = 'Double Proxy UCE for erasing concepts in Stable Diffusion')
    parser.add_argument('--edit_concepts', help='prompts corresponding to concepts to erase separated by ;', type=str, required=True)
    parser.add_argument('--guide_concepts', help='Concepts to guide the erased concepts towards seperated by ;', type=str, default=None)
    parser.add_argument('--preserve_concepts', help='Concepts to preserve seperated by ;', type=str, default=None)
    parser.add_argument('--concept_type', help='type of concept being erased', choices=['art', 'object'], type=str, required=True)
    parser.add_argument('--replace_indices', help='For each erase concept, specify which word indices to replace (0-indexed). Format: "all" or "0,1;all;1" (semicolon separates concepts, comma separates indices). "all" means replace all words. Default is "all" for all concepts.', type=str, default='all')
    
    parser.add_argument('--model_id', help='Model to run UCE on', type=str, default="CompVis/stable-diffusion-v1-4",)
    parser.add_argument('--device', help='cuda devices to train on', type=str, required=False, default='cuda:0')
    
    parser.add_argument('--expand_prompts', help='do you wish to expand your prompts?', choices=['true', 'false'], type=str, required=False, default='false')
    
    parser.add_argument('--save_dir', help='where to save your uce model weights', type=str, default='.')
    parser.add_argument('--exp_name', help='Use this to name your saved filename', type=str, default=None)
    
    args = parser.parse_args()
    
    device = args.device
    torch_dtype = torch.float32
    model_id = args.model_id
    
    concept_type = args.concept_type
    expand_prompts = args.expand_prompts
    
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    exp_name = args.exp_name
    if exp_name is None:
        exp_name = 'uce_double_proxy_test'

    # erase concepts
    edit_concepts = [concept.strip() for concept in args.edit_concepts.split(';')]
    
    # guide concepts
    guide_concepts = args.guide_concepts 
    if guide_concepts is None:
        guide_concepts = ''
        if concept_type == 'art':
            guide_concepts = 'art'
    guide_concepts = [concept.strip() for concept in guide_concepts.split(';')]
    # Remove empty strings from guide concepts
    guide_concepts = [c for c in guide_concepts if c]
    
    print(f"\nGuide concepts: {guide_concepts}")
    print(f"Edit concepts: {edit_concepts}")
    print(f"Note: All guide concepts will be used for all erase concepts\n")

    # preserve concepts
    if args.preserve_concepts is None:
        preserve_concepts = []
    else:
        preserve_concepts = [concept.strip() for concept in args.preserve_concepts.split(';')]
    
    # Parse replace_indices
    replace_indices = None
    if args.replace_indices is not None:
        replace_indices_str = args.replace_indices.split(';')
        replace_indices = []
        for idx_str in replace_indices_str:
            idx_str = idx_str.strip()
            if idx_str.lower() == 'all' or idx_str == '':
                replace_indices.append(None)  # None means replace all words
            else:
                # Parse comma-separated indices
                try:
                    indices = [int(i.strip()) for i in idx_str.split(',')]
                    replace_indices.append(indices)
                except ValueError:
                    raise ValueError(f"Invalid replace_indices format: '{idx_str}'. Use comma-separated integers or 'all'")
        
        # Ensure we have the right number of index specifications
        if len(replace_indices) != len(edit_concepts):
            raise ValueError(f"Number of replace_indices ({len(replace_indices)}) must match number of edit_concepts ({len(edit_concepts)})")
    
    # Expand prompts if requested
    if expand_prompts == 'true':
        edit_concepts_ = copy.deepcopy(edit_concepts)
        guide_concepts_ = copy.deepcopy(guide_concepts)

        for concept, guide_concept in zip(edit_concepts_, guide_concepts_):
            if concept_type == 'art':
                edit_concepts.extend([f'painting by {concept}',
                                       f'art by {concept}',
                                       f'artwork by {concept}',
                                       f'picture by {concept}',
                                       f'style of {concept}'
                                      ]
                                     )
                guide_concepts.extend([f'painting by {guide_concept}',
                                       f'art by {guide_concept}',
                                       f'artwork by {guide_concept}',
                                       f'picture by {guide_concept}',
                                       f'style of {guide_concept}'
                                      ]
                                     )

            else:
                edit_concepts.extend([f'image of {concept}',
                                       f'photo of {concept}',
                                       f'portrait of {concept}',
                                       f'picture of {concept}',
                                       f'painting of {concept}'
                                      ]
                                     )
                guide_concepts.extend([f'image of {guide_concept}',
                                       f'photo of {guide_concept}',
                                       f'portrait of {guide_concept}',
                                       f'picture of {guide_concept}',
                                       f'painting of {guide_concept}'
                                      ]
                                     )

    print(f"\n\n{'='*60}")
    print(f"Double Proxy UCE - Concept Erasing")
    print(f"{'='*60}")
    print(f"\nErasing: {edit_concepts}")
    print(f"Guiding: {guide_concepts}")
    print(f"Preserving: {preserve_concepts}\n")
    print(f"{'='*60}\n")
    
    pipe = DiffusionPipeline.from_pretrained(model_id, 
                                             torch_dtype=torch_dtype, 
                                             safety_checker=None).to(device)
    
    UCE_double_proxy(pipe, edit_concepts, guide_concepts, preserve_concepts, save_dir, exp_name, replace_indices)
