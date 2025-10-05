from os import path
import os
from sae_lens import SAE
from transformer_lens import HookedTransformer
from datasets import load_dataset
import torch
import sys
from utils import get_file_suffix

def main():
    dataset = "OpenWebText"
    hf_path = "vietgpt/openwebtext_en"
    model = "gemma-2-2b"
    sae_release = "gemma-scope-2b-pt-res-canonical"
    sae_id = "layer_24/width_16k/canonical"
    layer = 24
    hook_point = "blocks.24.hook_resid_post"
    k = 3
    batch_size = 512
    num_batches = 2000
    seq_len = 128
    total_num_tokens = num_batches * batch_size * seq_len

    # load
    ds = load_dataset(hf_path, split="train", streaming=True).iter(batch_size)
    model = HookedTransformer.from_pretrained_no_processing(model, device="cuda", dtype=torch.float32)
    sae = SAE.from_pretrained(sae_release, sae_id, device="cuda")[0]


    histogram = torch.zeros((sae.cfg.d_sae,), dtype=torch.long, device="cuda")
    for i in range(num_batches):
        texts  = next(ds)["text"]
        encoded_batch = model.tokenizer(texts, 
                                 padding="max_length",
                                 truncation=True,
                                 max_length=seq_len,
                                 return_tensors="pt")
        tokens = encoded_batch.input_ids.to("cuda")
        attention_mask = encoded_batch.attention_mask.to("cuda")
        
        pad_token_mask = (tokens == model.tokenizer.pad_token_id) # [batch, seq_len]

        # Get activations
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                attention_mask=attention_mask,
                names_filter=[hook_point],
                stop_at_layer=layer + 1,
            )
        acts = cache[hook_point]  # [batch, seq_len, d_model]
        
        # zero out pad token activations
        acts[pad_token_mask, :] = 0

        # Get dictionary activations
        with torch.no_grad():
            dict_acts = sae.encode(acts) # [batch, seq_len, d_sae]


        # Update histogram
        # topk = torch.topk(dict_acts, k=k, dim=-1)
        # topk_indices = topk.indices  # [batch, seq_len, k]
        # topk_indices = topk_indices.reshape(-1)
        # histogram += torch.bincount(topk_indices, minlength=sae.cfg.d_sae)

        # Update histogram with all non-zero indices
        nonzero_indices = torch.nonzero(dict_acts, as_tuple=True)[2] # [num_nonzero in batch]
        histogram += torch.bincount(nonzero_indices, minlength=sae.cfg.d_sae)
        print(f"Processed {i + 1}/{num_batches} batches", file=sys.stderr)

    dir_path = "checkpoints/general_histograms/gemma-2-2b_layer_24"
    os.makedirs(dir_path, exist_ok=True)
    file_name = "histogram_nonzero" + get_file_suffix(k=3, dataset=dataset, num_tokens=total_num_tokens) + ".pt"
    torch.save(histogram.cpu(), path.join(dir_path, file_name))
    print(f"Saving histogram to {path.join(dir_path, file_name)}", file=sys.stderr)