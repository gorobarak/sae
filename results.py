# %%
import os
os.environ["HF_HOME"] = "/home/yandex/APDL2425a/group_12/gorodissky/.cache/huggingface"
from sentence_transformers import SentenceTransformer
from datasets import load_dataset, load_from_disk, disable_progress_bar
import torch
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, DataCollatorWithPadding
from probes import eval_baseline, eval, create_dataset_baseline, project
# %%
model_to_colors = {
    # "meta-llama/Llama-3.1-8B-Instruct": "Orange",
    # "Qwen2.5-7B-Instruct": "Green",
    # "phi-3": "Red",
    "mistral-7b-instruct": "Blue",
}
# %%
# Vanilla experiment
task = "pred_nll"
baseline_model = "google/embeddinggemma-300m"
dataset_name = "allenai/WildChat-1M"
pooling_strategy = "last"
for model_name, color in model_to_colors.items():
    relative_error_baseline, r2_score_baseline = eval_baseline(baseline_model,
                                                               dataset_name,
                                                               model_name,
                                                               task)
    
    print(f"Evaluating model: {model_name}")
    layers, rel_errs, r2_scores = eval(model_name, task, dataset_name,pooling_strategy=pooling_strategy)
    layer_percentages = [layer / max(layers) for layer in layers]
    
    plt.plot(layer_percentages, rel_errs, marker='o', label=model_name, color=color)
    plt.axhline(y=relative_error_baseline, color=color, linestyle='--')

plt.xlabel("Layer percentage")
plt.ylabel("Relative Error")    
plt.grid()
plt.legend(loc='lower center', ncols=2, bbox_to_anchor=(0.5, -0.3))
plt.title(rf"{task} $(\downarrow)$")
plt.show()

# %%
# Projected dimension experiment 
task = "pred_nll"
baseline_model = "google/embeddinggemma-300m"
dataset_name = "allenai/WildChat-1M"
dims = [32, 64, 128, 256, 512]
for model, color in model_to_colors.items():
    for idx, dim in enumerate(dims):
        relative_error_baseline, r2_score_baseline = eval_baseline(baseline_model,
                                                                   dataset_name,
                                                                   model, 
                                                                   task, 
                                                                   project_dim=dim)
        print(f"Model: {model}, Dim: {dim}, Rel Error Baseline: {relative_error_baseline}, R2 Score Baseline: {r2_score_baseline}")

        print(f"Evaluating model: {model} with projected dim: {dim}")
        layers, rel_errs, r2_scores = eval(model, task, project_dim=dim)
        layer_percentages = [layer / max(layers) for layer in layers]
        # vary color shade based on dim
        intensity = 0.3 + 0.7 * ((idx + 1) / (len(dims)))
        plt.plot(layer_percentages, rel_errs, marker='o', label=f"{model} dim={dim}", color=color, alpha=intensity)
        plt.axhline(y=relative_error_baseline, color=color, linestyle='--', alpha=intensity)

    plt.xlabel("Layer percentage")
    plt.ylabel("Relative Error")    
    plt.grid()
    plt.legend(loc='lower center', ncols=2, bbox_to_anchor=(0.5, -0.4))
    plt.title(rf"{task} Relative Error $(\downarrow)$")
    plt.show()


# %%
# compare baselins 
baselines = ["google/embeddinggemma-300m"]
dataset_name = "allenai/WildChat-1M"
llm_models = ["meta-llama/Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct", "phi-3", "mistral-7b-instruct"]
tasks = ["pred_nll", "pred_tokens_mass"]
for task in tasks:
    print(f"Task: {task}")
    for llm_model in llm_models:
        print(f" {llm_model}")
        for baseline_model in baselines:
            rel_err, r2_score = eval_baseline(baseline_model, dataset_name, llm_model, task)
            if baseline_model == "google/embeddinggemma-300m":
                baseline_model = "EmbeddingGemma-300M"
            print(f"   {baseline_model}: err (lower): {rel_err:.2f}, r^2 (higher): {r2_score:.2f}")

    print("\n")    
# %%
# Baseline performance vs projected dimension
dims = [32, 64, 128, 256, 512]
baseline_model = "google/embeddinggemma-300m"
dataset_name = "allenai/WildChat-1M"
task = "pred_length"
for model in model_to_colors.keys():
    rel_errs, r2_scores = [], []
    for dim in dims:
        rel_err, r2_score = eval_baseline(baseline_model, dataset_name, model, task, project_dim=dim)
        rel_errs.append(rel_err)
        r2_scores.append(r2_score)

    plt.plot(dims, rel_errs, marker='o', label=model, color=model_to_colors[model])
plt.xlabel("Dimension")
plt.ylabel("Relative Error")
plt.grid()
plt.legend(loc='lower center', ncols=2, bbox_to_anchor=(0.5, -0.3))
plt.title(rf"Baseline performance {task} Relative Error $(\downarrow)$")
plt.show()
# %%
# Statistics of Y
Y_length_qwen = torch.load("data/pred_gen/allenai/WildChat-1M/Qwen2.5-7B-Instruct/Y_lengths.pt")
Y_length_qwen = Y_length_qwen[Y_length_qwen != -1]
Y_length_llama = torch.load("data/pred_gen/allenai/WildChat-1M/meta-llama/Llama-3.1-8B-Instruct/Y_lengths.pt")
Y_length_llama = Y_length_llama[Y_length_llama != -1]
Y_length_phi3 = torch.load("data/pred_gen/allenai/WildChat-1M/phi-3/Y_lengths.pt")
Y_length_phi3 = Y_length_phi3[Y_length_phi3 != -1]
Y_length_mistral = torch.load("data/pred_gen/allenai/WildChat-1M/mistral-7b-instruct/Y_lengths.pt")
Y_length_mistral = Y_length_mistral[Y_length_mistral != -1]


print(f"Qwen Length Y: mean={Y_length_qwen.float().mean():.2f}, std={Y_length_qwen.float().std():.2f}, ratio std/mean={Y_length_qwen.float().std()/Y_length_qwen.float().mean():.2f}")
print(f"Llama Length Y: mean={Y_length_llama.float().mean():.2f}, std={Y_length_llama.float().std():.2f}, ratio std/mean={Y_length_llama.float().std()/Y_length_llama.float().mean():.2f}")
print(f"phi-3 Length Y: mean={Y_length_phi3.float().mean():.2f}, std={Y_length_phi3.float().std():.2f}, ratio std/mean={Y_length_phi3.float().std()/Y_length_phi3.float().mean():.2f}")
print(f"mistral Length Y: mean={Y_length_mistral.float().mean():.2f}, std={Y_length_mistral.float().std():.2f}, ratio std/mean={Y_length_mistral.float().std()/Y_length_mistral.float().mean():.2f}")

# %%
# Pooling strategy comparison
task = "lengths"
dataset_name = "allenai/WildChat-1M"
baseline_model = "google/embeddinggemma-300m"
for model_name, color in model_to_colors.items():
    base_err, base_r2 = eval_baseline(baseline_model,
                                        dataset_name,
                                        model_name,
                                        task,
                                        project_dim=None)
    color_map = cm.get_cmap(color+"s")
    alphas = [0.3, 0.45, 0.6]
    for pooling_strategy, alpha in zip(["mean", "max", "last"], alphas): 
        
        print(f"Evaluating model: {model_name} with pooling: {pooling_strategy}")
        layers, rel_errs, r2_scores = eval(model_name,
                                           task,
                                           dataset_name,
                                           pooling_strategy=pooling_strategy)
                                           
        layer_percentages = [layer / max(layers) for layer in layers]
        color = color_map(alpha)
        plt.plot(layer_percentages, rel_errs, marker='o', label=f"{model_name} ({pooling_strategy})", color=color)
        plt.axhline(y=base_err, color=color, linestyle='--')
    plt.xlabel("Layer percentage")
    plt.ylabel("Relative Error %")
    plt.grid()
    plt.legend(loc='lower center', ncols=2, bbox_to_anchor=(0.5, -0.3))
    plt.title(rf"Pooling comparison on predicting {task} $(\downarrow)$")
    plt.show()
# %%
