import torch
from transformer_lens import HookedTransformer
from datasets import load_dataset
import os
import sys


# Load the model
model_name = "gemma-2-2b" 
dtype = torch.float32
model = HookedTransformer.from_pretrained(model_name, device="cuda", dtype=dtype)

# Load dataset
dataset = load_dataset("fancyzhx/dbpedia_14", split="train")


batch_size = 1024
seq_len = 128
hook_point = "blocks.18.hook_resid_post"
hook_layer = 18 
prepend_bos = True
pad_token_idx = model.tokenizer.pad_token_id 
reactivations = True
num_classes = 7  


NUM_SAMPELS_IN_CLASS = 40000

# Create activations
# iterate through the first 7 classes
for j in range(num_classes):
    activations= []
    offset = j * NUM_SAMPELS_IN_CLASS
    
    
    for i in range(offset, offset + NUM_SAMPELS_IN_CLASS, batch_size):
        interval_end = min(i + batch_size, offset + NUM_SAMPELS_IN_CLASS)
        batch = dataset[i: interval_end]
        tokens = model.to_tokens(batch['content'], truncate=True, move_to_device=True, prepend_bos=prepend_bos)
        tokens = tokens[:, :seq_len] # [batch, seq_len]
        if tokens.shape[-1] < seq_len:
            padding = torch.full((tokens.shape[0], seq_len - tokens.shape[-1]), pad_token_idx, device=tokens.device)
            tokens = torch.cat([tokens, padding], dim=-1)
        
        pad_token_mask = (tokens == pad_token_idx) # [batch, seq_len]
        
        if reactivations:
            tokens = torch.cat([tokens, tokens], dim=1) # [batch, seq_len * 2]
        
        
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=[hook_point],
                stop_at_layer=hook_layer + 1,  
            )
        curr_acts = cache[hook_point] # [batch, seq_len OR seq_len *2, d_model]
        if reactivations:
            curr_acts = curr_acts[:, seq_len:, :] # [batch, seq_len, d_model] Take only the second half of the activations if reactivations is True
        curr_acts[pad_token_mask, :] = 0.0 # Zero out activations for padding tokens
        activations.append(curr_acts.cpu()) # move to CPU to preserve GPU memory


    activations_tensor = torch.cat(activations, dim=0)  
    

    dir_path = os.path.join("checkpoints", f"{model_name}_layer_{hook_layer}", f"dbpedia_class_{j}")
    os.makedirs(dir_path, exist_ok=True)
    file_name = "reactivations.pt" if reactivations else "activations.pt"
    with open(os.path.join(dir_path, file_name), "wb") as f:
        torch.save(activations_tensor, f)
    
    print(f"Activations saved to {os.path.join(dir_path, file_name)}", file=sys.stderr)
