
import os
import sys
os.environ["HF_HOME"] = "/home/yandex/APDL2425a/group_12/gorodissky/.cache/huggingface"
from sentence_transformers import SentenceTransformer
from  probes import create_dataset_perplexity, create_baseline_dataset
import torch
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
from datasets import load_dataset


dataset = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
def keep_only_query(example: dict) -> dict:
    new_conv = []
    new_conv.append(example["conversation"][0])  # keep only the first message
    return {"conversation": new_conv} 
dataset = dataset.map(keep_only_query)
TL_model_names = ["qwen2.5-0.5b-instruct", "qwen2.5-7b-instruct", "meta-llama/Llama-3.1-8B-Instruct", "gemma-2-2b-it"]
hf_model_names = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", "google/gemma-2-2b-it"]

tokenizers = [AutoTokenizer.from_pretrained(name) for name in hf_model_names] 
models = [HookedTransformer.from_pretrained_no_processing(name) for name in TL_model_names]
for model, model_name, tokenizer in zip(models, TL_model_names, tokenizers):
    print(f"Processing model: {model_name}", file=sys.stderr)
    Xs, Y = create_dataset_perplexity(
        model,
        tokenizer,
        dataset,
        dataset_size=1000)

    dir_path = f"data/pred_perplexity/{model_name}"
    for layer_idx, X in enumerate(Xs):
        if (layer_idx == 1):
            print(f"X shape : {X.shape}", file=sys.stderr)
            print(f"Y shape : {Y.shape}", file=sys.stderr)
        os.makedirs(f"{dir_path}/blocks.{layer_idx}.hook_resid_post", exist_ok=True)
        torch.save(X, f"{dir_path}/blocks.{layer_idx}.hook_resid_post/X.pt")
        torch.save(Y, f"{dir_path}/blocks.{layer_idx}.hook_resid_post/Y.pt")
# sentence_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# embeddings = create_baseline_dataset(
#     sentence_model,
#     dataset,
#     dataset_size=1000)
# print(f"Embeddings shape: {embeddings.shape}")
# os.makedirs("data/all-mpnet-base-v2/allenai/WildChat-1M", exist_ok=True)
# torch.save(embeddings, "data/all-mpnet-base-v2/allenai/WildChat-1M/embeddings.pt") 