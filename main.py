
import os
os.environ["HF_HOME"] = "/home/yandex/APDL2425a/group_12/gorodissky/.cache/huggingface"
import sys
from sentence_transformers import SentenceTransformer
from  probes import create_dataset_lengths, create_dataset_perplexity, create_dataset_tokens_mass, code_words
import torch
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
from datasets import load_dataset


task_names = ["pred_perplexity", "pred_tokens_mass", "pred_length"]
dataset_name = "allenai/WildChat-1M"
for task_name in task_names:
    print(f"Creating dataset for task: {task_name}", file=sys.stderr)
    TL_model_names = ["meta-llama/Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct", "phi-3", "mistral-7b-instruct"] 
    hf_model_names = ["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "microsoft/Phi-3-mini-4k-instruct", "mistralai/Mistral-7B-Instruct-v0.1"] 
    tokenizers = [AutoTokenizer.from_pretrained(name) for name in hf_model_names] 

    for tl_model_name, hf_model_name, tokenizer in zip(TL_model_names, hf_model_names, tokenizers):
        print(f"Processing model: {tl_model_name}", file=sys.stderr)
        model = HookedTransformer.from_pretrained_no_processing(tl_model_name, device='cpu')
        
        if task_name == "pred_tokens_mass":
            token_ids = sum(tokenizer(code_words, add_special_tokens=False)["input_ids"], [])
        
        if task_name == "pred_tokens_mass":
            Xs, Y = create_dataset_tokens_mass(
            model,
            hf_model_name,
            tokenizer,
            dataset_name,
            token_ids)
        
        elif task_name == "pred_perplexity":
            Xs, Y = create_dataset_perplexity(
                model,
                hf_model_name,
                tokenizer,
                dataset_name)
        else:
            Xs, Y = create_dataset_lengths(
                model,
                hf_model_name,
                tokenizer,
                dataset_name)

        dir_path = f"data/{task_name}/{tl_model_name}"
        for layer_idx, X in enumerate(Xs):
            if (layer_idx == 0):
                print(f"X shape : {X.shape}", file=sys.stderr)
                print(f"Y shape : {Y.shape}", file=sys.stderr)
            os.makedirs(f"{dir_path}/blocks.{layer_idx}.hook_resid_post", exist_ok=True)
            torch.save(X, f"{dir_path}/blocks.{layer_idx}.hook_resid_post/X.pt")
            torch.save(Y, f"{dir_path}/blocks.{layer_idx}.hook_resid_post/Y.pt")
        
        del model
        del tokenizer
        Xs.clear()
        del Y
# sentence_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# embeddings = create_baseline_dataset(
#     sentence_model,
#     dataset,
#     dataset_size=1000)
# print(f"Embeddings shape: {embeddings.shape}")
# os.makedirs("data/all-mpnet-base-v2/allenai/WildChat-1M", exist_ok=True)
# torch.save(embeddings, "data/all-mpnet-base-v2/allenai/WildChat-1M/embeddings.pt") 