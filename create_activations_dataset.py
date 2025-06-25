import torch
from transformer_lens import HookedTransformer
from datasets import load_dataset
import os


# Load the model
model_name = "gemma-2-2b"
model = HookedTransformer.from_pretrained(model_name, device="cuda", dtype=torch.float32)

# Load dataset
dataset = load_dataset("fancyzhx/dbpedia_14", split="train")

# Create activations
batch_size = 500
seq_len = 128
hook_point = "blocks.20.hook_resid_post"
hook_layer = 20  

NUM_SAMPELS_IN_CLASS = 40000


# iterate through the 14 classes
for j in range(0,14):
    
    activations= []
    offset = j * NUM_SAMPELS_IN_CLASS
    
    for i in range(offset, offset + NUM_SAMPELS_IN_CLASS, batch_size):
        batch = dataset[i: i + batch_size]
        tokens = model.to_tokens(batch['content'], truncate=True, move_to_device=True, prepend_bos=False)
        tokens = tokens[:, :seq_len] # [batch, seq_len]
        if tokens.shape[-1] < seq_len:
            pad_token_idx = model.cfg.d_vocab - 1
            padding = torch.full((tokens.shape[0], seq_len - tokens.shape[1]), pad_token_idx, device=tokens.device)
            tokens = torch.cat([tokens, padding], dim=1)
        
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=[hook_point],
                stop_at_layer=hook_layer + 1,  
            )
        curr_acts = cache[hook_point] # [batch, seq_len, d_model]
        activations.append(curr_acts)


    activations_tensor = torch.cat(activations, dim=0).cpu()  # move to CPU so when loaded is on CPU
    dir_name = f"dbpedia_class_{j}_{model_name}_{hook_point}"
    dir_path = os.path.join("checkpoints", dir_name)
    os.makedirs(dir_path, exist_ok=True)

    with open(os.path.join(dir_path, "activations.pkl"), "wb") as f:
        torch.save(activations_tensor, f)
    
    print(f"Activations saved to {os.path.join(dir_path, 'activations.pkl')}")
# Save activations to disk