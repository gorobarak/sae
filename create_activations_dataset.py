import torch
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_attention_mask
from datasets import load_dataset
import os
import sys


def main():
    # Load the model
    model_name = "gemma-2-2b" 
    dtype = torch.float32
    model = HookedTransformer.from_pretrained_no_processing(model_name, device="cuda", dtype=dtype)

    # Load dataset
    dataset = load_dataset("fancyzhx/dbpedia_14", split="train")


    batch_size = 1024
    seq_len = 128
    hook_point = "blocks.24.hook_resid_post"
    hook_layer = int(hook_point.split(".")[1])  # Extract layer number from hook point 
    pad_token_idx = model.tokenizer.pad_token_id 
    reactivations = False
    num_classes = 7  
    NUM_SAMPLES_IN_CLASS = 40000
    num_of_classes_in_dbpedia = 14

    # Create activations
    for j in range(7, num_of_classes_in_dbpedia):
        activations= []
        offset = j * NUM_SAMPLES_IN_CLASS
        
        for i in range(offset, offset + NUM_SAMPLES_IN_CLASS, batch_size):
            interval_end = min(i + batch_size, offset + NUM_SAMPLES_IN_CLASS)
            batch = dataset[i: interval_end]
            
            encoded_batch = model.tokenizer(batch['content'], 
                                            padding="max_length", 
                                            truncation=True,
                                            max_length=seq_len,
                                            return_tensors="pt")  # [batch, seq_len]
            tokens = encoded_batch.input_ids
            attention_mask = encoded_batch.attention_mask
            
            pad_token_mask = (tokens == pad_token_idx) # [batch, seq_len]
            
            if reactivations:
                tokens = torch.cat([tokens, tokens], dim=1) # [batch, seq_len * 2]
                attention_mask = torch.cat([attention_mask, attention_mask], dim=1) # [batch, seq_len * 2]
                

            with torch.no_grad():
                _, cache = model.run_with_cache(
                    tokens,
                    attention_mask=attention_mask,
                    names_filter=[hook_point],
                    stop_at_layer=hook_layer + 1,  
                )
            curr_acts = cache[hook_point] # [batch, seq_len OR seq_len *2, d_model]
            
            if reactivations:
                curr_acts = curr_acts[:, seq_len:, :] # [batch, seq_len, d_model] Take only the second half of the activations if reactivations is True
            
            curr_acts[pad_token_mask, :] = 0.0 # Zero out activations for padding tokens
            
            assert curr_acts.shape[0] == interval_end - i
            assert curr_acts.shape[1] == seq_len
            assert curr_acts.shape[2] == model.cfg.d_model
            assert len(curr_acts.shape) == 3
            
            activations.append(curr_acts.cpu()) # move to CPU to preserve GPU memory


        activations_tensor = torch.cat(activations, dim=0)  
        dir_path = os.path.join("checkpoints", f"{model_name}_layer_{hook_layer}", f"dbpedia_class_{j}")
        os.makedirs(dir_path, exist_ok=True)
        file_name = "reactivations.pt" if reactivations else "activations.pt"
        torch.save(activations_tensor, os.path.join(dir_path, file_name))
        print(f"Activations saved to {os.path.join(dir_path, file_name)}", file=sys.stderr)
