import torch
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_attention_mask
from datasets import load_dataset
import os
import sys
from utils import get_dataset_metadata, prepare_batch_texts, get_label_column


def main(dataset_name):
    # Load the model
    model_name = "gemma-2-2b" 
    dtype = torch.float32
    model = HookedTransformer.from_pretrained_no_processing(model_name, device="cuda", dtype=dtype)

    # Load dataset and sort by label
    num_classes, class_to_size, hf_path = get_dataset_metadata(dataset_name)
    dataset = load_dataset(hf_path, split="train")
    label_column = get_label_column(dataset_name)
    dataset = dataset.sort(label_column)

    batch_size = 1024
    seq_len = 128
    hook_point = "blocks.24.hook_resid_post"
    hook_layer = int(hook_point.split(".")[1])  # Extract layer number from hook point 
    pad_token_idx = model.tokenizer.pad_token_id 
    reactivations = False
    

    # Create activations
    class_size_prefix_sum = 0
    for j in range(0, num_classes):
        activations= []
        cur_class_size = class_to_size[j]

        for i in range(class_size_prefix_sum, class_size_prefix_sum + cur_class_size, batch_size):
            interval_end = min(i + batch_size, class_size_prefix_sum + cur_class_size)
            batch = dataset[i: interval_end]
            texts = prepare_batch_texts(batch, dataset_name)

            encoded_batch = model.tokenizer(texts, 
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

        class_size_prefix_sum += cur_class_size

        activations_tensor = torch.cat(activations, dim=0)  
        dir_path = os.path.join("checkpoints", f"{model_name}_layer_{hook_layer}", f"{dataset_name}", f"class_{j}")
        os.makedirs(dir_path, exist_ok=True)
        file_name = "reactivations.pt" if reactivations else "activations.pt"
        torch.save(activations_tensor, os.path.join(dir_path, file_name))
        print(f"Saved activations to {os.path.join(dir_path, file_name)}", file=sys.stderr)

