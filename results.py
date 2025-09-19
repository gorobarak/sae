#%%
import os
os.environ["HF_HOME"] = "/home/yandex/APDL2425a/group_12/gorodissky/.cache/huggingface"
import torch
import pandas as pd
import matplotlib.pyplot as plt
import wandb
# %%
api = wandb.Api()
wandb_project = "dbpedia_classification"
title = "DBPEDIA"
for run in api.runs(f"gorodissky-tel-aviv-university/{wandb_project}"):
    name = run.name
    print(run.created_at)
    if name== "sae_insights":
        sae_insights_df = run.history(pandas=True)
    elif name == "sae_insights_non_private":
        sae_insights_non_private_df = run.history(pandas=True)
    elif name == "dense_representation":
        dense_representation_df= run.history(pandas=True)
    elif name == "dense_representation_non_private":
        dense_representation_non_private_df = run.history(pandas=True)
    elif name == "sae_insights_TFIDF":
        sae_insights_tfidf_df = run.history(pandas=True)
    elif name == "sae_insights_TFIDF_non_private":
        sae_insights_tfidf_non_private_df = run.history(pandas=True)
    elif name == "sae_insights_shuffled_DP_sensitivity_False":
        without_sensitivity = run.history(pandas=True)
    elif name == "sae_insights_shuffled_DP_sensitivity_True":
        with_sensitivity = run.history(pandas=True)

print("sae insights df:")
print(sae_insights_df)
print("dense representation df:")
print(dense_representation_df)
print("sae insights non private df:")
print(sae_insights_non_private_df)
print("dense representation non private df:")
print(dense_representation_non_private_df)
print("sae insights tfidf df:")
print(sae_insights_tfidf_df)
print("sae insights tfidf non private df:")
print(sae_insights_tfidf_non_private_df)
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


# SI TFIDF
color = "green"
axes[0].plot(sae_insights_tfidf_df["epsilon"], sae_insights_tfidf_df["top1_acc"], 
             marker='o', label="sae_insights_TFIDF", color=color)
lower = sae_insights_tfidf_df["top1_acc"] - sae_insights_tfidf_df["top1_acc_std"]
upper = sae_insights_tfidf_df["top1_acc"] + sae_insights_tfidf_df["top1_acc_std"]
axes[0].fill_between(sae_insights_tfidf_df["epsilon"], lower, upper, color=color, alpha=0.15)
axes[0].axhline(sae_insights_tfidf_non_private_df["top1_acc"].item(), 
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


# SI TFIDF
color = "green"
axes[1].plot(sae_insights_tfidf_df["epsilon"], sae_insights_tfidf_df["top3_acc"], 
             marker='o', label="sae_insights_TFIDF", color=color)
lower = sae_insights_tfidf_df["top3_acc"] - sae_insights_tfidf_df["top3_acc_std"]
upper = sae_insights_tfidf_df["top3_acc"] + sae_insights_tfidf_df["top3_acc_std"]
axes[1].fill_between(sae_insights_tfidf_df["epsilon"], lower, upper, color=color, alpha=0.15)
axes[1].axhline(sae_insights_tfidf_non_private_df["top3_acc"].item(), 
                linestyle='--', color=color)

axes[1].set_xlabel("Epsilon")
# axes[1].set_xscale("log")
axes[1].set_ylabel("Top-3 Accuracy")
axes[1].set_title("Top-3 Accuracy vs Epsilon")
axes[1].legend()

fig.grid()
fig.suptitle(title)
fig.tight_layout()
# %%
### Compare shuffled DP with/without sensitivity 
datasets = ["yahoo_questions", "yahoo_answers", "ag_news", "dbpedia"]
fig, axes = plt.subplots(4,2, figsize=(10, 15))
axes = axes.flat
api = wandb.Api()
for i, dataset_name in enumerate(datasets):
    wandb_project = dataset_name + "_classification"
    title = dataset_name
    for run in api.runs(f"gorodissky-tel-aviv-university/{wandb_project}"):
        name = run.name
        if name== "sae_insights_TFIDF_non_private":
            sae_insights_non_private_df = run.history(pandas=True)
        elif name == "sae_insights_shuffled_DP_sensitivity_False":
            without_sensitivity = run.history(pandas=True)
        elif name == "sae_insights_shuffled_DP_sensitivity_True":
            with_sensitivity = run.history(pandas=True)
    # TOP1
    color = "royalblue"
    line_without = axes[2*i].plot(without_sensitivity["epsilon"], without_sensitivity["top1_acc"], 
                marker='o', label="without_sensitivity", color=color)[0]
    lower_bound = without_sensitivity["top1_acc"] - without_sensitivity["top1_acc_std"]
    upper_bound = without_sensitivity["top1_acc"] + without_sensitivity["top1_acc_std"]
    axes[2*i].fill_between(without_sensitivity["epsilon"], lower_bound, upper_bound, color=color, alpha=0.15)


    color = "purple"
    line_with = axes[2*i].plot(with_sensitivity["epsilon"], with_sensitivity["top1_acc"], 
                marker='o', label="with_sensitivity", color=color)[0]
    lower_bound = with_sensitivity["top1_acc"] - with_sensitivity["top1_acc_std"]
    upper_bound = with_sensitivity["top1_acc"] + with_sensitivity["top1_acc_std"]
    axes[2*i].fill_between(with_sensitivity["epsilon"], lower_bound, upper_bound, color=color, alpha=0.15)


    axes[2*i].axhline(sae_insights_non_private_df["top1_acc"].item(), linestyle='--', color="black")
    axes[2*i].set_xlabel("Epsilon")
    axes[2*i].set_ylabel("Top-1 Accuracy")
    axes[2*i].set_title(title)
    axes[2*i].grid()

    # TOP3
    color = "royalblue"
    axes[2*i+1].plot(without_sensitivity["epsilon"], without_sensitivity["top3_acc"], 
                marker='o', label="without_sensitivity", color=color)
    lower_bound = without_sensitivity["top3_acc"] - without_sensitivity["top3_acc_std"]
    upper_bound = without_sensitivity["top3_acc"] + without_sensitivity["top3_acc_std"]
    axes[2*i+1].fill_between(without_sensitivity["epsilon"], lower_bound, upper_bound, color=color, alpha=0.15)

    color = "purple"
    axes[2*i+1].plot(with_sensitivity["epsilon"], with_sensitivity["top3_acc"], 
                marker='o', label="with_sensitivity", color=color)
    lower_bound = with_sensitivity["top3_acc"] - with_sensitivity["top3_acc_std"]
    upper_bound = with_sensitivity["top3_acc"] + with_sensitivity["top3_acc_std"]
    axes[2*i+1].fill_between(with_sensitivity["epsilon"], lower_bound, upper_bound, color=color, alpha=0.15)

    axes[2*i+1].axhline(sae_insights_non_private_df["top3_acc"].item(), linestyle='--', color="black")
    axes[2*i+1].set_xlabel("Epsilon")
    axes[2*i+1].set_ylabel("Top-3 Accuracy")
    axes[2*i+1].set_title(title)
    axes[2*i+1].grid()

fig.suptitle("Shuffled DP: with vs without sensitivity (TFIDF used)")
fig.legend(handles=[line_without, line_with], loc="lower center", ncols=2)
fig.tight_layout()
# %%
### Compare sae_insights with TFIDF vs without TFIDF
datasets = ["yahoo_questions", "yahoo_answers", "ag_news", "dbpedia"]
fig, axes = plt.subplots(4,2, figsize=(10, 15))
axes = axes.flat
api = wandb.Api()
for i, dataset_name in enumerate(datasets):
    wandb_project = dataset_name + "_classification"
    title = dataset_name
    for run in api.runs(f"gorodissky-tel-aviv-university/{wandb_project}"):
        name = run.name
        if name== "sae_insights_non_private":
            non_private_without_tfidf = run.history(pandas=True)
        elif name == "sae_insights_TFIDF_non_private":
            non_private_with_tfidf = run.history(pandas=True)
        elif name == "sae_insights":
            without_tfidf = run.history(pandas=True)
        elif name == "sae_insights_TFIDF":
            with_tfidf = run.history(pandas=True)
    # TOP1
    color = "red"
    line_without = axes[2*i].plot(without_tfidf["epsilon"], without_tfidf["top1_acc"], 
                marker='o', label="without_TFIDF", color=color)[0]
    lower_bound = without_tfidf["top1_acc"] - without_tfidf["top1_acc_std"]
    upper_bound = without_tfidf["top1_acc"] + without_tfidf["top1_acc_std"]
    axes[2*i].fill_between(without_tfidf["epsilon"], lower_bound, upper_bound, color=color, alpha=0.15)
    axes[2*i].axhline(non_private_without_tfidf["top1_acc"].item(), linestyle='--', color=color)



    color = "green"
    line_with = axes[2*i].plot(with_tfidf["epsilon"], with_tfidf["top1_acc"], 
                marker='o', label="with_TFIDF", color=color)[0]
    lower_bound = with_tfidf["top1_acc"] - with_tfidf["top1_acc_std"]
    upper_bound = with_tfidf["top1_acc"] + with_tfidf["top1_acc_std"]
    axes[2*i].fill_between(with_tfidf["epsilon"], lower_bound, upper_bound, color=color, alpha=0.15)
    axes[2*i].axhline(non_private_with_tfidf["top1_acc"].item(), linestyle='--', color=color)


    axes[2*i].set_xlabel("Epsilon")
    axes[2*i].set_ylabel("Top-1 Accuracy")
    axes[2*i].set_title(title)
    axes[2*i].grid()

    # TOP3
    color = "red"
    axes[2*i+1].plot(without_tfidf["epsilon"], without_tfidf["top3_acc"], 
                marker='o', label="without_TFIDF", color=color)
    lower_bound = without_tfidf["top3_acc"] - without_tfidf["top3_acc_std"]
    upper_bound = without_tfidf["top3_acc"] + without_tfidf["top3_acc_std"]
    axes[2*i+1].fill_between(without_tfidf["epsilon"], lower_bound, upper_bound, color=color, alpha=0.15)
    axes[2*i+1].axhline(non_private_without_tfidf["top3_acc"].item(), linestyle='--', color=color)

    color = "green"
    axes[2*i+1].plot(with_tfidf["epsilon"], with_tfidf["top3_acc"], 
                marker='o', label="with_TFIDF", color=color)
    lower_bound = with_tfidf["top3_acc"] - with_tfidf["top3_acc_std"]
    upper_bound = with_tfidf["top3_acc"] + with_tfidf["top3_acc_std"]
    axes[2*i+1].fill_between(with_tfidf["epsilon"], lower_bound, upper_bound, color=color, alpha=0.15)
    axes[2*i+1].axhline(non_private_with_tfidf["top3_acc"].item(), linestyle='--', color=color)
    

    axes[2*i+1].set_xlabel("Epsilon")
    axes[2*i+1].set_ylabel("Top-3 Accuracy")
    axes[2*i+1].set_title(title)
    axes[2*i+1].grid()

fig.suptitle("SAE Insights: with vs without TFIDF")
fig.legend(handles=[line_without, line_with], loc="lower center", ncols=2)
fig.tight_layout()
# %%