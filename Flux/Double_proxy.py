import torch
torch.set_grad_enabled(False)
import argparse
import os
import copy
import time
import gc

from safetensors.torch import save_file
from diffusers import DiffusionPipeline


def UCE_double_proxy(model_id, edit_concepts, guide_concepts, preserve_concepts,
                     save_dir, exp_name, torch_dtype, device, max_sequence_length,
                     replace_indices=None):
    """
    Double Proxy Solution for Concept Erasing in FLUX

    The Double Proxy algorithm:
      Step 1: Compute proxy vector v_i* as the projection of v_i onto the span of guide concepts
      Step 2: Find minimal weight update ΔW that maps v_i → v_i*, constrained so that
              preserve concept outputs are unchanged.

    IMPORTANT — nn.Linear convention:
      PyTorch nn.Linear stores weight W with shape [out_features=p, in_features=n].
      The forward pass computes:  output = input @ W^T
      In matrix-math notation this is equivalent to  W @ input^T  (transposed).
      Therefore, for input token embedding c  (shape [n]):
          nn.Linear output = W @ c   (viewed as [p,1])  where W has shape [p,n]
      All operations below follow the [p,n] convention (same as SD Double_proxy.py).
    """

    # ── 1. Load only the transformer (no VAE/text encoders) to get the modules ──
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        vae=None,
        text_encoder=None,
        text_encoder_2=None,
        tokenizer=None,
        tokenizer_2=None,
        torch_dtype=torch_dtype,
        safety_checker=None,
    )

    uce_modules = []
    uce_module_names = []
    for name, module in pipe.transformer.named_modules():
        if 'context_embedder' in name or 'text_embedder.linear_1' in name:
            if hasattr(module, 'weight') and module.weight is not None:
                uce_modules.append(module.to(device))
                uce_module_names.append(name)

    original_modules = copy.deepcopy(uce_modules)
    uce_modules = copy.deepcopy(uce_modules)

    pipe = None
    torch.cuda.empty_cache()
    gc.collect()

    # ── 2. Load text encoders only to obtain token embeddings ──
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        vae=None,
        transformer=None,
        torch_dtype=torch_dtype,
        safety_checker=None,
    ).to(device)

    start_time = time.time()

    all_concepts = list(dict.fromkeys(edit_concepts + guide_concepts + preserve_concepts))

    # uce_erase_embeds[concept] = { 'T5': list of [n,1] tensors, 'CLIP': [p,1] tensor }
    uce_erase_embeds = {}
    print("\nCollecting text embeddings ...")
    for e in all_concepts:
        if not e or e in uce_erase_embeds:
            continue

        t_emb = pipe.encode_prompt(
            prompt=e,
            prompt_2=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
        )
        # t_emb[0]: T5 sequence embeddings  [1, seq_len, 4096]
        # t_emb[1]: CLIP pooled embedding   [1, 768]

        # T5: collect one embedding per real token (skip first/last BOS/EOS)
        attn_mask = pipe.tokenizer_2(
            e,
            padding="max_length",
            max_length=max_sequence_length,
            return_overflowing_tokens=False,
            truncation=True,
            return_length=False,
            return_tensors="pt",
        )['attention_mask']
        valid_token_count = attn_mask.sum().item()

        # FLUX T5 tokenizer (T5-v1.1) has NO BOS token.
        # Token layout: [content_0, content_1, ..., content_k, EOS, PAD, PAD, ...]
        # valid_token_count = k+2 (content tokens + EOS).
        # We want indices 0 … valid_token_count-2  (all content, skip EOS).
        t5_token_embeds = []
        for tok_idx in range(0, int(valid_token_count) - 1):
            t5_token_embeds.append(t_emb[0][:, tok_idx:tok_idx + 1, :])  # [1, 1, 4096]

        # CLIP: single pooled vector [1, 768]
        clip_embed = t_emb[1]  # [1, 768]

        uce_erase_embeds[e] = {'T5': t5_token_embeds, 'CLIP': clip_embed}
        print(f"  '{e}': {len(t5_token_embeds)} T5 token(s), CLIP pooled [{clip_embed.shape[-1]}]")

    pipe = None
    torch.cuda.empty_cache()
    gc.collect()

    # ── 3. Initialise replace_indices ──
    if replace_indices is None:
        replace_indices = [None] * len(edit_concepts)
    if len(replace_indices) != len(edit_concepts):
        raise ValueError("replace_indices length must match edit_concepts length")

    # ── 4. Double Proxy per module ──
    for module_idx, module in enumerate(original_modules):
        W_0 = module.weight.clone()  # [p, n]

        # Determine which embedding type to use for this module
        # context_embedder: n = 4096 (T5)
        # text_embedder.linear_1: n = 768 (CLIP pooled)
        use_clip = (W_0.shape[1] == 768)
        emb_key = 'CLIP' if use_clip else 'T5'
        print(f"\nModule [{module_idx}] {uce_module_names[module_idx]} | shape {list(W_0.shape)} | using {emb_key}")

        def get_vecs(concept):
            """Return list of float32 column vectors [n,1] for a concept."""
            entry = uce_erase_embeds.get(concept)
            if entry is None:
                return []
            if use_clip:
                # CLIP pooled: single vector; cast to float32 for numerical stability
                return [entry['CLIP'].squeeze(0).unsqueeze(-1).to(device=device, dtype=torch.float32)]  # [n,1]
            else:
                # T5 tokens: list of vectors; cast to float32
                return [tok.squeeze(0).squeeze(0).unsqueeze(-1).to(device=device, dtype=torch.float32)
                        for tok in entry['T5']]  # each [n,1]

        # Work in float32 throughout; convert W_0 once and cast back at the end.
        W_0_f32 = W_0.float()  # [p, n]

        for erase_idx, erase_concept in enumerate(edit_concepts):
            erase_vecs = get_vecs(erase_concept)
            if not erase_vecs:
                print(f"  Skipping '{erase_concept}': no embeddings found")
                continue

            # Which token indices to process
            indices_to_replace = replace_indices[erase_idx]
            if indices_to_replace is None:
                tokens_to_process = list(range(len(erase_vecs)))
            else:
                tokens_to_process = [i for i in indices_to_replace
                                     if 0 <= i < len(erase_vecs)]

            for token_idx in tokens_to_process:
                v_i = erase_vecs[token_idx]  # [n, 1] float32

                # ── Step 1: Proxy c_i* = proj_S(c_i) — Eq.(2) in paper ──
                # S = matrix whose columns span the safe subspace (guide concepts).
                # c_i* = S (S^T S)^+ S^T c_i
                # Single guide (k=1): collapses to UCE — set c_i* = s_1 (the guide embedding).
                guide_vecs = []
                for gc_name in guide_concepts:
                    guide_vecs.extend(get_vecs(gc_name))

                if not guide_vecs:
                    print(f"  Warning: no guide vectors for '{erase_concept}' token {token_idx}, skipping.")
                    continue

                if len(guide_vecs) == 1:
                    # UCE-equivalent: use guide embedding directly as proxy
                    v_i_star = guide_vecs[0]  # [n,1] float32
                else:
                    # General projection: c_i* = S (S^T S)^+ S^T c_i
                    V_g = torch.cat(guide_vecs, dim=1)  # [n, k]
                    VtV_pinv = torch.linalg.pinv(V_g.T @ V_g)  # [k,k] float32
                    v_i_star = V_g @ VtV_pinv @ (V_g.T @ v_i)  # [n,1]

                # ── Step 2: ΔW constrained to left null-space of C_pres ──
                # Constraint: ΔW C_pres = 0  ⟹  ΔW = Z U_2^T
                # where U_2 ∈ R^{n x (n-r)} is orthonormal basis for left null-space of C_pres.
                # Left null-space of C_pres = right null-space of C_pres^T.
                # SVD of C_pres^T  [m,n]: Vh rows from rank onwards are the right null-space vectors.
                preserve_vecs = []
                for pc in preserve_concepts:
                    preserve_vecs.extend(get_vecs(pc))

                if not preserve_vecs:
                    # Unconstrained update: ΔW c_i = b = W_0(c_i* - c_i)
                    b = W_0_f32 @ (v_i_star - v_i)  # [p,1]
                    denom = (v_i.T @ v_i).item()
                    if denom < 1e-10:
                        continue
                    Delta_W = (b @ v_i.T) / denom  # [p, n]
                    W_0_f32 = W_0_f32 + Delta_W
                    print(f"  Unconstrained update for '{erase_concept}' token {token_idx}")
                    continue

                # C_pres: [n, m] — stack all preserve concept token embeddings column-wise
                V_p = torch.cat(preserve_vecs, dim=1)  # [n, m] float32

                # SVD of C_pres^T [m, n] → Vh shape [n, n]
                _, S_svd, Vh = torch.linalg.svd(V_p.T, full_matrices=True)
                rank_V = int((S_svd > 1e-6).sum().item())
                # U_2: columns span left null-space of C_pres; shape [n, n-r]
                U_2 = Vh[rank_V:, :].T  # [n, n-r] float32

                # x = U_2^T c_i  ∈ R^{n-r}   (feasible direction on c_i)
                # b = W_0 (c_i* - c_i)  ∈ R^p  (desired output shift)
                x = U_2.T @ v_i                          # [n-r, 1] float32
                b = W_0_f32 @ (v_i_star - v_i)           # [p, 1]   float32

                x_norm_sq = (x.T @ x).item()
                if x_norm_sq < 1e-10:
                    print(f"  Warning: ||x||^2={x_norm_sq:.2e} too small, skipping token {token_idx}")
                    continue

                # Z* = b x^T / ||x||^2,  ΔW* = Z* U_2^T  — Eqs.(4,5) in paper
                Z = (b @ x.T) / x_norm_sq   # [p, n-r]
                Delta_W = Z @ U_2.T          # [p, n]
                W_0_f32 = W_0_f32 + Delta_W
                print(f"  Updated '{erase_concept}' token {token_idx} | "
                      f"||ΔW||={Delta_W.norm().item():.4f}  ||x||²={x_norm_sq:.4e}")

        # Cast back to the model's dtype
        W_0 = W_0_f32.to(torch_dtype)

        uce_modules[module_idx].weight = torch.nn.Parameter(W_0)

    os.makedirs(save_dir, exist_ok=True)
    uce_state_dict = {
        name + '.weight': module.weight
        for name, module in zip(uce_module_names, uce_modules)
    }
    save_path = os.path.join(save_dir, exp_name + '.safetensors')
    save_file(uce_state_dict, save_path)

    end_time = time.time()
    print(f'\n\nErased concepts using Double Proxy (FLUX)\n'
          f'Model edited in {end_time - start_time:.2f} seconds\n'
          f'Weights saved to {save_path}\n')


# ── CLI ─────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='DoubleProxyFlux',
        description='Double Proxy concept erasure for FLUX')
    parser.add_argument('--edit_concepts', type=str, required=True,
                        help='Concepts to erase, separated by ;')
    parser.add_argument('--guide_concepts', type=str, default=None,
                        help='Guide concepts separated by ;')
    parser.add_argument('--preserve_concepts', type=str, default=None,
                        help='Concepts to preserve, separated by ;')
    parser.add_argument('--concept_type', type=str, required=True,
                        choices=['art', 'object'])
    parser.add_argument('--replace_indices', type=str, default='all',
                        help='Per-concept token replacement indices. '
                             'Format: "all" or "0,1;all;1". "all" replaces every token. '
                             'Semicolons separate concepts; commas separate indices.')

    parser.add_argument('--model_id', type=str,
                        default='black-forest-labs/FLUX.1-schnell')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--save_dir', type=str, default='./models')
    parser.add_argument('--exp_name', type=str, default=None)

    parser.add_argument('--expand_prompts', type=str, default='false',
                        choices=['true', 'false'])

    args = parser.parse_args()

    device = args.device
    torch_dtype = torch.bfloat16
    model_id = args.model_id

    max_sequence_length = 256 if 'schnell' in model_id else 512

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    exp_name = args.exp_name or 'flux_double_proxy_test'

    edit_concepts = [c.strip() for c in args.edit_concepts.split(';')]

    guide_concepts = args.guide_concepts or ('art' if args.concept_type == 'art' else '')
    guide_concepts = [c.strip() for c in guide_concepts.split(';') if c.strip()]

    preserve_concepts = (
        [c.strip() for c in args.preserve_concepts.split(';')]
        if args.preserve_concepts else []
    )

    # Parse replace_indices
    replace_indices = None
    if args.replace_indices:
        parts = args.replace_indices.split(';')
        replace_indices = []
        for p in parts:
            p = p.strip()
            if p.lower() == 'all' or p == '':
                replace_indices.append(None)
            else:
                replace_indices.append([int(i.strip()) for i in p.split(',')])
        if len(replace_indices) != len(edit_concepts):
            raise ValueError('replace_indices length must match edit_concepts length')

    # Expand prompts
    if args.expand_prompts == 'true':
        import copy as _copy
        ec_, gc_ = _copy.deepcopy(edit_concepts), _copy.deepcopy(guide_concepts)
        for c, g in zip(ec_, gc_):
            if args.concept_type == 'art':
                for tpl in [f'painting by {c}', f'art by {c}', f'artwork by {c}',
                            f'picture by {c}', f'style of {c}']:
                    edit_concepts.append(tpl)
                for tpl in [f'painting by {g}', f'art by {g}', f'artwork by {g}',
                            f'picture by {g}', f'style of {g}']:
                    guide_concepts.append(tpl)
            else:
                for tpl in [f'image of {c}', f'photo of {c}', f'portrait of {c}',
                            f'picture of {c}', f'painting of {c}']:
                    edit_concepts.append(tpl)
                for tpl in [f'image of {g}', f'photo of {g}', f'portrait of {g}',
                            f'picture of {g}', f'painting of {g}']:
                    guide_concepts.append(tpl)

    print(f"\nErasing  : {edit_concepts}")
    print(f"Guiding  : {guide_concepts}")
    print(f"Preserving: {preserve_concepts}\n")

    UCE_double_proxy(
        model_id=model_id,
        edit_concepts=edit_concepts,
        guide_concepts=guide_concepts,
        preserve_concepts=preserve_concepts,
        save_dir=save_dir,
        exp_name=exp_name,
        torch_dtype=torch_dtype,
        device=device,
        max_sequence_length=max_sequence_length,
        replace_indices=replace_indices,
    )
