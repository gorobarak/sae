#%%
import os
os.environ["HF_HOME"] = "/home/yandex/APDL2425a/group_12/gorodissky/.cache/huggingface"
import torch
import pandas as pd
import matplotlib.pyplot as plt
import wandb
import torch.distributions as distributions
# %%
api = wandb.Api()
wandb_project = "dbpedia_classification"
title = "DBPEDIA"
for run in api.runs(f"gorodissky-tel-aviv-university/{wandb_project}"):
    name = run.name
    if name== "sae_insights":
        sae_insights_df = run.history(pandas=True)
    elif name == "sae_insights_non_private":
        sae_insights_non_private_df = run.history(pandas=True)
    elif name == "dense_representation":
        dense_representation_df= run.history(pandas=True)
    elif name == "dense_representation_non_private":
        dense_representation_non_private_df = run.history(pandas=True)

print("sae insights df:")
print(sae_insights_df)
print("dense representation df:")
print(dense_representation_df)
print("sae insights non private df:")
print(sae_insights_non_private_df)
print("dense representation non private df:")
print(dense_representation_non_private_df)
# %%
fig, axes = plt.subplots(1, 2)

# TOP1 acc
# DR
color = "blue"
axes[0].plot(dense_representation_df["epsilon"], 
             dense_representation_df["top1_acc"], 
             marker='o', label="dense_representation", color=color)
lower = dense_representation_df["top1_acc"] - dense_representation_df["top1_acc_std"]
upper = dense_representation_df["top1_acc"] + dense_representation_df["top1_acc_std"]
axes[0].fill_between(dense_representation_df["epsilon"], lower, upper, color=color, alpha=0.15)
axes[0].axhline(dense_representation_non_private_df["top1_acc"].item(), 
                linestyle='--', color=color)

# SI
color = "orange"
axes[0].plot(sae_insights_df["epsilon"], sae_insights_df["top1_acc"], 
             marker='o', label="sae_insights", color=color)
lower = sae_insights_df["top1_acc"] - sae_insights_df["top1_acc_std"]
upper = sae_insights_df["top1_acc"] + sae_insights_df["top1_acc_std"]
axes[0].fill_between(sae_insights_df["epsilon"], lower, upper, color=color, alpha=0.15)
axes[0].axhline(sae_insights_non_private_df["top1_acc"].item(), 
                linestyle='--', color=color)


axes[0].set_xlabel("Epsilon")
# axes[0].set_xscale("log")
axes[0].set_ylabel("Top-1 Accuracy")
axes[0].set_title("Top-1 Accuracy vs Epsilon")
axes[0].legend()

# TOP3 acc
# DR
color = "blue"
axes[1].plot(dense_representation_df["epsilon"], dense_representation_df["top3_acc"], 
             marker='o', label="dense_representation", color=color)
lower = dense_representation_df["top3_acc"] - dense_representation_df["top3_acc_std"]
upper = dense_representation_df["top3_acc"] + dense_representation_df["top3_acc_std"]
axes[1].fill_between(dense_representation_df["epsilon"], lower, upper, color=color, alpha=0.15)
axes[1].axhline(dense_representation_non_private_df["top3_acc"].item(), 
                linestyle='--', color=color)

# SI
color = "orange"
axes[1].plot(sae_insights_df["epsilon"], sae_insights_df["top3_acc"], marker='o', label="sae_insights", color=color)
lower = sae_insights_df["top3_acc"] - sae_insights_df["top3_acc_std"]
upper = sae_insights_df["top3_acc"] + sae_insights_df["top3_acc_std"]
axes[1].fill_between(sae_insights_df["epsilon"], lower, upper, color=color, alpha=0.15)
axes[1].axhline(sae_insights_non_private_df["top3_acc"].item(), 
                linestyle='--', color=color)

axes[1].set_xlabel("Epsilon")
# axes[1].set_xscale("log")
axes[1].set_ylabel("Top-3 Accuracy")
axes[1].set_title("Top-3 Accuracy vs Epsilon")
axes[1].legend()


fig.suptitle(title)
fig.tight_layout()
# %%




# %%
