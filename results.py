# %%
import os
from tkinter import Y
os.environ["HF_HOME"] = "/home/yandex/APDL2425a/group_12/gorodissky/.cache/huggingface"
from sentence_transformers import SentenceTransformer
from datasets import load_dataset, load_from_disk, disable_progress_bar
import torch
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
import matplotlib.pyplot as plt
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, DataCollatorWithPadding
from probes import eval_baseline, eval, create_dataset_baseline, project
# %%
model_to_colors = {
    "meta-llama/Llama-3.1-8B-Instruct": "orange",
    "Qwen2.5-7B-Instruct": "green",
    # "phi-3": "red",
    # "mistral-7b-instruct": "blue",
}
# %%
task = "pred_length"
baseline_model = "google/embeddinggemma-300m"
for model_name, color in model_to_colors.items():
    relative_error_baseline, r2_score_baseline = eval_baseline(baseline_model, model_name, task)
    
    print(f"Evaluating model: {model_name}")
    layers, rel_errs, r2_scores = eval(model_name, task)
    layer_percentages = [layer / max(layers) for layer in layers]
    
    plt.plot(layer_percentages, rel_errs, marker='o', label=model_name, color=color)
    plt.axhline(y=relative_error_baseline, color=color, linestyle='--')

plt.xlabel("Layer percentage")
plt.ylabel("Relative Error")    
plt.grid()
plt.legend(loc='lower center', ncols=2, bbox_to_anchor=(0.5, -0.3))
plt.title(rf"{task} Relative Error $(\downarrow)$")
plt.show()

# %%
task = "pred_tokens_mass"
baseline_model = "google/embeddinggemma-300m"
dims = [32, 64, 128, 256, 512]
for model, color in model_to_colors.items():
    for idx, dim in enumerate(dims):
        relative_error_baseline, r2_score_baseline = eval_baseline(baseline_model, model, task, project_dim=dim)
        print(f"Model: {model}, Dim: {dim}, Rel Error Baseline: {relative_error_baseline}, R2 Score Baseline: {r2_score_baseline}")

        print(f"Evaluating model: {model} with projected dim: {dim}")
        layers, rel_errs, r2_scores = eval(model, task, project_dim=dim)
        layer_percentages = [layer / max(layers) for layer in layers]
        # vary color shade based on dim
        intensity = 0.3 + 0.7 * (idx / (len(dims)))
        plt.plot(layer_percentages, rel_errs, marker='o', label=f"{model} dim={dim}", color=color, alpha=intensity)
        plt.axhline(y=relative_error_baseline, color=color, linestyle='--', alpha=intensity)

    plt.xlabel("Layer percentage")
    plt.ylabel("Relative Error")    
    plt.grid()
    plt.legend(loc='lower center', ncols=2, bbox_to_anchor=(0.5, -0.4))
    plt.title(rf"{task} Relative Error $(\downarrow)$")
    plt.show()


# %%
# compare baseline 
baselines = ["all-mpnet-base-v2", "google/embeddinggemma-300m"]
llm_models = ["meta-llama/Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct", "phi-3", "mistral-7b-instruct"]
tasks = ["pred_perplexity", "pred_tokens_mass"]
for task in tasks:
    print(f"Task: {task}")
    for llm_model in llm_models:
        print(f" {llm_model}")
        for baseline_model in baselines:
            rel_err, r2_score = eval_baseline(baseline_model, llm_model, task)
            if baseline_model == "google/embeddinggemma-300m":
                baseline_model = "EmbeddingGemma-300M"
            print(f"   {baseline_model}: err (lower): {rel_err:.2f}, r^2 (higher): {r2_score:.2f}")

    print("\n")    
# %%
dims = [2, 4, 8 ,16 ,32, 64, 128, 256, 512,  600, 700, None, 800, 1024, 1500]
baseline_model = "all-mpnet-base-v2"
task = "pred_perplexity"
for model in model_to_colors.keys():
    print(f"{model}")
    for dim in dims:
        rel_err, r2_score = eval_baseline(baseline_model, model, task, project_dim=dim)
        print(f"   Dim {dim}: err (lower): {rel_err:.2f}, r^2 (higher): {r2_score:.2f}")
# %%
Y_llama = torch.log(torch.load("data/pred_perplexity/meta-llama/Llama-3.1-8B-Instruct/blocks.0.hook_resid_post/Y.pt") + 1e-10)
Y_qwen = torch.log(torch.load("data/pred_perplexity/Qwen2.5-7B-Instruct/blocks.0.hook_resid_post/Y.pt") + 1e-10)
Y_mistral = torch.log(torch.load("data/pred_perplexity/mistral-7b-instruct/blocks.0.hook_resid_post/Y.pt") + 1e-10)
Y_phi = torch.log(torch.load("data/pred_perplexity/phi-3/blocks.0.hook_resid_post/Y.pt") + 1e-10)

# print std and mean
print("LLama: mean ", torch.mean(Y_llama).item(), " std ", torch.std(Y_llama).item())
print("Qwen: mean ", torch.mean(Y_qwen).item(), " std ", torch.std(Y_qwen).item())
print("Mistral: mean ", torch.mean(Y_mistral).item(), " std ", torch.std(Y_mistral).item())
print("Phi: mean ", torch.mean(Y_phi).item(), " std ", torch.std(Y_phi).item())
# %%

gemma_2_2b = HookedTransformer.from_pretrained_no_processing("gemma-2-2b")
W_U = gemma_2_2b.W_U
print(W_U.shape)
# %%
