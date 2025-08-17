#%%
import os
from random import sample
os.environ["HF_HOME"] = "/home/yandex/APDL2425a/group_12/gorodissky/.cache/huggingface"
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import torch
from transformer_lens import HookedTransformer
import pandas as pd
import matplotlib.pyplot as plt
import wandb
import torch.distributions as distributions
# %%
api = wandb.Api()

for run in api.runs("gorodissky-tel-aviv-university/privacy_experiment_whole_class_summarization"):
    name = run.name
    if name== "features_histogram":
        feature_hist_df = run.history(pandas=True)
    elif name == "dense_representation":
        dense_representation_df= run.history(pandas=True)

print("features histogram df:")
print(feature_hist_df)
print("dense representation df:")
print(dense_representation_df)
# %%
fig, axes = plt.subplots(1, 2)

# TOP1 acc
axes[0].plot(dense_representation_df["epsilon"], dense_representation_df["top1_acc"], marker='o', label="dense_representation")
axes[0].plot(feature_hist_df["epsilon"], feature_hist_df["top1_acc"], marker='o', label="features_histogram")
axes[0].set_xlabel("Epsilon")
axes[0].set_xscale("log")
axes[0].set_ylabel("Top-1 Accuracy")
axes[0].set_title("Top-1 Accuracy vs Epsilon")
axes[0].legend()

# TOP3 acc
axes[1].plot(dense_representation_df["epsilon"], dense_representation_df["top3_acc"], marker='o', label="dense_representation")
axes[1].plot(feature_hist_df["epsilon"], feature_hist_df["top3_acc"], marker='o', label="features_histogram")
axes[1].set_xlabel("Epsilon")
axes[1].set_xscale("log")
axes[1].set_ylabel("Top-3 Accuracy")
axes[1].set_title("Top-3 Accuracy vs Epsilon")
axes[1].legend()

fig.tight_layout()
# %%
dense_representation_df
# %%
laplace = distributions.Laplace(loc=0, scale=3840)
# %%
sample = laplace.sample((100000,))
# %%
sample.mean()
# %%
sample.quantile(0.9)
# %%
