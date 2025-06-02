# %%
from torch import topk
import wandb
import pandas as pd
import pickle

api = wandb.Api()

project = "final_with_accuracy"
entity = "gorodissky-tel-aviv-university"

# %%
runs = api.runs(f"{entity}/{project}")
data = []

for run in runs:
    #config_sae_type = run.config.get("sae_type", None)
    #dict_size = run.config.get("dict_size", None)
    # k = run.config.get("top_k", None)
    # final_l0_norm = run.summary.get("l0_norm", None)
    # final_l2_loss = run.summary.get("l2_loss", None)
    # final_ce_degradation = run.summary.get("performance/ce_degradation", None)
    # data.append({
    #     "config_sae_type": config_sae_type,
    #     "dictionary_size": dict_size,
    #     "k": k,
    #     "l0_norm": final_l0_norm,
    #     "normalized_mse": final_l2_loss,
    #     "ce_degradation": final_ce_degradation
    # })
    baseline = run.config.get("baseline", None)
    fine_tune = run.config.get("fine_tune", None)
    aggregate_function = run.config.get("aggregate_function", None)
    
    final_ce_loss = run.summary.get("ce_loss", None)
    final_accuracy = run.summary.get("accuracy", None)
    
    data.append({
        "basline": baseline,
        "fine_tune": fine_tune,
        "aggregate_function": aggregate_function,
        "final_ce_loss": final_ce_loss,
        "final_accuracy": final_accuracy
    })

df = pd.DataFrame(data)
print(df)

# %%
def label(row):
    label = ""
    if row['basline'] == True:
        label += "Baseline"
    else:
        label += "SAE"
    if row['fine_tune'] == True:
        label += " + Fine-tune"
    else:
        label += " + No Fine-tune"
    if row['aggregate_function'] == "max":
        label += " + Max"
    elif row['aggregate_function'] == "mean":
        label += " + Mean"
    return label
df['label'] = df.apply(label, axis=1)
print(df)
# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# %%
ce_losses = df['final_ce_loss'].values
labels = df['label'].values
x = range(len(labels))  # the label locations
# plt.figure(figsize=(12, 6))
plt.scatter(x, ce_losses, s=50, c=['red', 'red', 'blue', 'blue', 'red', 'red', 'blue', 'blue'], alpha=0.6)
plt.xticks(x, labels, rotation=45, ha='right')
plt.xlabel('Configuration')
plt.ylabel('Final CE Loss')
plt.grid(True)

# fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# # Plot 1: Normalized MSE vs Dictionary size (k=32)
# for sae_type in ['batchtopk', 'topk']:
#     data = df[(df['config_sae_type'] == sae_type) & (df['k'] == 32.)]
#     data = data.sort_values(by='dictionary_size')
#     axs[0, 0].plot(data['dictionary_size'], data['normalized_mse'], 
#                    marker='o', linestyle='--', label=f"{sae_type} (k=32)")

# axs[0, 0].set_title('Normalized MSE vs Dictionary size (k=32)')
# axs[0, 0].set_xlabel('Dictionary size')
# axs[0, 0].set_ylabel('Normalized MSE')
# axs[0, 0].set_xscale('log')
# axs[0, 0].legend()
# axs[0, 0].grid(True)

# # Plot 2: Normalized MSE vs k (Dict size = 12288)
# for sae_type in ['batchtopk', 'topk', 'jumprelu']:
#     data = df[(df['config_sae_type'] == sae_type) & (df['dictionary_size'] == 12288)]
#     data = data.sort_values(by='dictionary_size')
#     axs[0, 1].plot(data['l0_norm'], data['normalized_mse'], 
#                    marker='o', linestyle='--', label=sae_type)

# axs[0, 1].set_title('Normalized MSE vs k (Dict size = 12288)')
# axs[0, 1].set_xlabel('k')
# axs[0, 1].set_ylabel('Normalized MSE')
# axs[0, 1].legend()
# axs[0, 1].grid(True)

# # Plot 3: CE degradation vs Dictionary size (k=32)
# for sae_type in ['batchtopk', 'topk']:
#     data = df[(df['config_sae_type'] == sae_type) & (df['k'] == 32)]
#     axs[1, 0].plot(data['dictionary_size'], data['ce_degradation'], 
#                    marker='o', linestyle='--', label=f"{sae_type} (k=32)")

# axs[1, 0].set_title('CE degradation vs Dictionary size (k=32)')
# axs[1, 0].set_xlabel('Dictionary size')
# axs[1, 0].set_ylabel('CE degradation')
# axs[1, 0].set_xscale('log')
# axs[1, 0].legend()
# axs[1, 0].grid(True)

# # Plot 4: CE degradation vs k (Dict size = 12288)
# for sae_type in ['batchtopk', 'topk', 'jumprelu']:
#     data = df[(df['config_sae_type'] == sae_type) & (df['dictionary_size'] == 12288)]
#     axs[1, 1].plot(data['l0_norm'], data['ce_degradation'], 
#                    marker='o', linestyle='--', label=sae_type)

# axs[1, 1].set_title('CE degradation vs k (Dict size = 12288)')
# axs[1, 1].set_xlabel('k')
# axs[1, 1].set_ylabel('CE degradation')
# axs[1, 1].legend()
# axs[1, 1].grid(True)

# Adjust layout and display the plot
# plt.tight_layout()
plt.show()
# %%
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from query_gpt import standerdize


# %%
with open("checkpoints/topk_samples_base/heaps_390.pkl", "rb") as f:
    topk_dict_base =  pickle.load(f)
with open("checkpoints/topk_samples_duplicate/heaps_390.pkl", "rb") as f:
    topk_dict_duplicate =  pickle.load(f)
with open("checkpoints/topk_samples_base/descriptions.pkl", "rb") as f:
    descriptions_base = pickle.load(f)
with open("checkpoints/topk_samples_duplicate/descriptions.pkl", "rb") as f:
    descriptions_duplicate = pickle.load(f)




# %%
def compare_descriptions(latent_id, neuropedia_desc):
    print((f"Latent {latent_id} comparison:"))
    print("--------------")
    print(f"Neuropedia description: {neuropedia_desc}")
    print("--------------") 
    print(f"Base description:\n{descriptions_base[latent_id]}")
    print("--------------")
    print(f"Duplicate description:\n{descriptions_duplicate[latent_id]}")

# %%
def compare_topk_samples(latent_id):
    print(f"Top 10 samples for latent {latent_id} (base):")
    heap_base = standerdize(topk_dict_base[latent_id])
    for sample in heap_base:
        print(sample)

    print("--------------")
    
    print(f"Top 10 samples for latent {latent_id} (duplicate):")
    heap_duplicate = standerdize(topk_dict_duplicate[latent_id])
    for sample in heap_duplicate:
        print(sample)
# %%
compare_descriptions(4162, "positive sentiments or mentions")
# %%
compare_topk_samples(4162)
# %%
compare_descriptions(6822, "expressions of positivity and positive sentiment")
# %%
compare_topk_samples(6822)
# %%
compare_descriptions(13563, "negative descriptors related to unfortunate or distressing situations")
# %%
compare_topk_samples(13563)
# %%
