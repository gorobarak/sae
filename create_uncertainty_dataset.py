import torch
from transformer_lens import HookedTransformer
import pickle
from datasets import load_dataset
from sae_lens import SAE
import sys

# parameters
dataset = "vietgpt/openwebtext_en"
model_name = "gemma-2-2b"
sae_release = "gemma-scope-2b-pt-res-canonical"
sae_id = "layer_24/width_16k/canonical"
hook_point = "blocks.24.hook_resid_post"
k = 3
max_generation_length = 128
num_examples = 10000
device = "cuda"
dtype = torch.float32

def main():
    # load
    ds = iter(load_dataset(dataset, split="train", streaming=True))
    model = HookedTransformer.from_pretrained_no_processing(model_name,
                                                            device=device,
                                                            dtype=dtype)
    sae = SAE.from_pretrained(sae_release,
                            sae_id,
                            device=device)[0]
    tokenizer = model.tokenizer

    examples = []
    for i in range(num_examples):
        prompt = next(ds)["text"]
        encoded_prompt = tokenizer(prompt,
                                   return_tensors="pt",
                                   truncation=True,
                                   max_length=128)
        tokens = encoded_prompt.input_ids.to(device)
        # update prompt after tunncation
        prompt = tokenizer.decode(tokens.squeeze(), skip_special_tokens=True)

        # generate continuation
        log_probs = [] # log probs of generated tokens
        acts_list = [] # activations of generated tokens
        for _ in range(max_generation_length):
            with torch.no_grad():
                logits, cache = model.run_with_cache(
                    tokens,
                    names_filter=[hook_point]
                )
            
            # greedy sampling
            probs = torch.softmax(logits[0,-1, :], dim=-1) # [vocab_size]
            next_token = probs.argmax() # Greedy generation
            tokens = torch.cat([tokens, next_token.unsqueeze(0).unsqueeze(0)], dim=1) 
            
            # save log prob
            next_token_log_prob = torch.log(probs[next_token])
            log_probs.append(next_token_log_prob.item())


            # Save activation
            last_token_act = cache[hook_point][0, -1, :]
            acts_list.append(last_token_act)

            if next_token == tokenizer.eos_token_id:
                break

        
        # create example
        # perplexity
        avg_nll = -torch.tensor(log_probs).mean() 
        perplexity = torch.exp(avg_nll).item()

        # sae features histogram
        acts = torch.stack(acts_list, dim=0) # [gen_seq_len, d_model]
        with torch.no_grad():
            sae_acts = acts @ sae.W_enc # [gen_seq_len, d_sae]
        topk = torch.topk(sae_acts, k=k, dim=-1) # [gen_seq_len, k]
        topk_indices = topk.indices # [gen_seq_len, k]
        topk_indices = topk_indices.reshape(-1) # [gen_seq_len * k]
        histogram = torch.bincount(topk_indices, minlength=sae.cfg.d_sae) # [d_sae]
        histogram = histogram.tolist()
        generation = tokenizer.decode(tokens.squeeze(), skip_special_tokens=True)
        generation = generation[len(prompt):]
        examples.append({
            "prompt": prompt,
            "generation": generation,
            "perplexity": perplexity,
            "sae_histogram": histogram
        })
        print(f"Created example {i+1}/{num_examples}", file=sys.stderr)

    # save 
    path = f"checkpoints/uncertainty_dataset_{dataset}_{num_examples}.pkl"
    with open(path, "wb") as f:
        pickle.dump(examples, f)
    print("Saved to uncertainty_dataset.pkl", file=sys.stderr)