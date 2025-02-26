import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm 
from scipy.sparse import csr_matrix, save_npz, vstack, load_npz
import gc
import scipy.stats as stats
import os
from sae_lens import SAE, HookedSAETransformer


def sae(model, model_sae, prompts):
    with torch.no_grad():
        _, cache = model.run_with_cache_with_saes(
                    prompts,
                    saes=[model_sae],
                    stop_at_layer=model_sae.cfg.hook_layer + 1,
                )

    result = cache[f"{model_sae.cfg.hook_name}.hook_sae_acts_post"].detach().cpu().numpy()
    
    # Garbage collection / cache reset
    del cache  
    torch.cuda.empty_cache()  
    torch.cuda.ipc_collect() 
    gc.collect()  
    
    return result


def process_batch(model, model_sae, batch):
    feats = sae(model, model_sae, batch)

    B, seq_len, feature_dim = feats.shape
    seq_lengths = [feats[j].shape[0] for j in range(B)]
    feats_2d = feats.reshape(B * seq_len, feature_dim)

    sparse_feats = csr_matrix(feats_2d)
    
    return sparse_feats, seq_lengths


def process_prompts(model, model_sae, prompts, batch_size=8):
    batches = [prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)]

    all_sparse, seq_lengths = [], []

    for batch in tqdm(batches):
        sparse_feats, batch_seq_lengths = process_batch(model, model_sae, batch)
        all_sparse.append(sparse_feats)
        seq_lengths.extend(batch_seq_lengths)
    
    final_sparse = vstack(all_sparse)

    return final_sparse, seq_lengths


def get_sae_embeddings(input_list, gemma_scope_sae_release, gemma_scope_sae_id, output_dir=None, filename=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    gemma_2_2b = HookedSAETransformer.from_pretrained_no_processing("gemma-2-2b", device=device, torch_dtype=torch.float16)

    gemma_2_2b_sae = SAE.from_pretrained(gemma_scope_sae_release, gemma_scope_sae_id, device=device)[0]

    sae_embeddings, seq_lengths = process_prompts(gemma_2_2b, gemma_2_2b_sae, input_list, batch_size=8)
    
    if output_dir:
        # Generate filename if not provided
        if filename is None:
            filename = "sae_embeddings"

        save_npz(f"../output/{filename}.npz", sae_embeddings)
        np.save(f"../output/{filename}_seq_lengths.npy", np.array(seq_lengths, dtype=object))

    return sae_embeddings, seq_lengths