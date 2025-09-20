#%%
import os
os.environ["HF_HOME"] = "/home/yandex/APDL2425a/group_12/gorodissky/.cache/huggingface"
from cv2 import line
import torch
import pandas as pd
import matplotlib.pyplot as plt
import wandb

# %%
### Central DP comprarison
datasets = ["yahoo_questions", "yahoo_answers", "ag_news", "dbpedia"]
fig, axes = plt.subplots(4,2, figsize=(10, 15))
axes = axes.flat
api = wandb.Api()
for i, dataset_name in enumerate(datasets):
    wandb_project = dataset_name + "_classification"
    title = dataset_name
    for run in api.runs(f"gorodissky-tel-aviv-university/{wandb_project}"):
        name = run.name
        if name == "sae_insights_TFIDF":
            sae_insights_df = run.history(pandas=True)
        elif name == "sae_insights_TFIDF_non_private":
            sae_insights_non_private_df = run.history(pandas=True)
        elif name == "dense_representation":
            dense_representation_df = run.history(pandas=True)
        elif name == "dense_representation_non_private":
            dense_representation_non_private_df = run.history(pandas=True)
    
    # TOP1
    color = "royalblue"
    line_sae = axes[2*i].plot(sae_insights_df["epsilon"], sae_insights_df["top1_acc"], 
                marker='o', label="sae_insights", color=color)[0]
    lower_bound = sae_insights_df["top1_acc"] - sae_insights_df["top1_acc_std"]
    upper_bound = sae_insights_df["top1_acc"] + sae_insights_df["top1_acc_std"]
    axes[2*i].fill_between(sae_insights_df["epsilon"], lower_bound, upper_bound, color=color, alpha=0.15)
    axes[2*i].axhline(sae_insights_non_private_df["top1_acc"].item(), linestyle='--', color=color)

    color = "orchid"
    line_dense = axes[2*i].plot(dense_representation_df["epsilon"], dense_representation_df["top1_acc"], 
                marker='o', label="dense_representation", color=color)[0]
    lower_bound = dense_representation_df["top1_acc"] - dense_representation_df["top1_acc_std"]
    upper_bound = dense_representation_df["top1_acc"] + dense_representation_df["top1_acc_std"]
    axes[2*i].fill_between(dense_representation_df["epsilon"], lower_bound, upper_bound, color=color, alpha=0.15)
    axes[2*i].axhline(dense_representation_non_private_df["top1_acc"].item(), linestyle='--', color=color)
    
    axes[2*i].set_xlabel("Epsilon")
    axes[2*i].set_ylabel("Top-1 Accuracy")
    axes[2*i].set_title(title)
    axes[2*i].grid()

    # TOP3
    color = "royalblue"
    axes[2*i+1].plot(sae_insights_df["epsilon"], sae_insights_df["top3_acc"], 
                marker='o', label="sae_insights", color=color)
    lower_bound = sae_insights_df["top3_acc"] - sae_insights_df["top3_acc_std"]
    upper_bound = sae_insights_df["top3_acc"] + sae_insights_df["top3_acc_std"]
    axes[2*i+1].fill_between(sae_insights_df["epsilon"], lower_bound, upper_bound, color=color, alpha=0.15)
    axes[2*i+1].axhline(sae_insights_non_private_df["top3_acc"].item(), linestyle='--', color=color)

    color = "orchid"
    axes[2*i+1].plot(dense_representation_df["epsilon"], dense_representation_df["top3_acc"], 
                marker='o', label="dense_representation", color=color)
    lower_bound = dense_representation_df["top3_acc"] - dense_representation_df["top3_acc_std"]
    upper_bound = dense_representation_df["top3_acc"] + dense_representation_df["top3_acc_std"]
    axes[2*i+1].fill_between(dense_representation_df["epsilon"], lower_bound, upper_bound, color=color, alpha=0.15)
    axes[2*i+1].axhline(dense_representation_non_private_df["top3_acc"].item(), linestyle='--', color=color)
    
    axes[2*i+1].set_xlabel("Epsilon")
    axes[2*i+1].set_ylabel("Top-3 Accuracy")
    axes[2*i+1].set_title(title)
    axes[2*i+1].grid()
fig.suptitle("Central DP: SAE Insights (TFIDF) vs Dense Representation")
fig.legend(handles=[line_sae, line_dense], loc="lower center", ncols=2)
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
# Shuffled DP and central DP comparison:
datasets = ["yahoo_questions", "yahoo_answers", "ag_news", "dbpedia"]
fig, axes = plt.subplots(4,2, figsize=(10, 15))
axes = axes.flat
api = wandb.Api()
for i, dataset_name in enumerate(datasets):
    wandb_project = dataset_name + "_classification"
    title = dataset_name
    for run in api.runs(f"gorodissky-tel-aviv-university/{wandb_project}"):
        name = run.name
        if name == "sae_insights_shuffled_DP_sensitivity_True":
            sae_insights_shuffled_dp_df = run.history(pandas=True)
        elif name == "sae_insights_TFIDF":
            sae_insights_df_central_dp = run.history(pandas=True)
        elif name == "sae_insights_TFIDF_non_private":
            sae_insights_non_private_df = run.history(pandas=True)
        elif name == "dense_representation_shuffled_DP":
            dense_representation_shuffled_dp_df = run.history(pandas=True)
        elif name == "dense_representation":
            dense_representation_df_central_dp = run.history(pandas=True)
        elif name == "dense_representation_non_private":
            dense_representation_non_private_df = run.history(pandas=True)
    
    # TOP1
    # SAE INSIGHTS shuffled DP
    color = "orchid"
    line_sae_shuffled_DP = axes[2*i].plot(sae_insights_shuffled_dp_df["epsilon"], sae_insights_shuffled_dp_df["top1_acc"], 
                marker='o', label="sae_insights_shuffled_DP", color=color)[0]
    lower_bound = sae_insights_shuffled_dp_df["top1_acc"] - sae_insights_shuffled_dp_df["top1_acc_std"]
    upper_bound = sae_insights_shuffled_dp_df["top1_acc"] + sae_insights_shuffled_dp_df["top1_acc_std"]
    axes[2*i].fill_between(sae_insights_shuffled_dp_df["epsilon"], lower_bound, upper_bound, color=color, alpha=0.15)

    # SAE INSIGHTS central DP
    color = "purple"
    line_sae_central = axes[2*i].plot(sae_insights_df_central_dp["epsilon"], sae_insights_df_central_dp["top1_acc"], 
                marker='o', label="sae_insights_central_DP", color=color)[0]
    lower_bound = sae_insights_df_central_dp["top1_acc"] - sae_insights_df_central_dp["top1_acc_std"]
    upper_bound = sae_insights_df_central_dp["top1_acc"] + sae_insights_df_central_dp["top1_acc_std"]
    axes[2*i].fill_between(sae_insights_df_central_dp["epsilon"], lower_bound, upper_bound, color=color, alpha=0.15)

    axes[2*i].axhline(sae_insights_non_private_df["top1_acc"].item(), linestyle='--', color=color)

    # DENSE REPRESENTATION shuffled DP
    color = "lightseagreen"
    line_dense_shuffled_DP = axes[2*i].plot(dense_representation_shuffled_dp_df["epsilon"], dense_representation_shuffled_dp_df["top1_acc"], 
                marker='o', label="dense_representation_shuffled_DP", color=color)[0]
    lower_bound = dense_representation_shuffled_dp_df["top1_acc"] - dense_representation_shuffled_dp_df["top1_acc_std"]
    upper_bound = dense_representation_shuffled_dp_df["top1_acc"] + dense_representation_shuffled_dp_df["top1_acc_std"]
    axes[2*i].fill_between(dense_representation_shuffled_dp_df["epsilon"], lower_bound, upper_bound, color=color, alpha=0.15)


    # DENSE REPRESENTATION central DP
    color = "royalblue"
    line_dense_central = axes[2*i].plot(dense_representation_df_central_dp["epsilon"], dense_representation_df_central_dp["top1_acc"], 
                marker='o', label="dense_representation_central_DP", color=color)[0]
    lower_bound = dense_representation_df_central_dp["top1_acc"] - dense_representation_df_central_dp["top1_acc_std"]
    upper_bound = dense_representation_df_central_dp["top1_acc"] + dense_representation_df_central_dp["top1_acc_std"]
    axes[2*i].fill_between(dense_representation_df_central_dp["epsilon"], lower_bound, upper_bound, color=color, alpha=0.15)

    axes[2*i].axhline(dense_representation_non_private_df["top1_acc"].item(), linestyle='--', color=color)

    axes[2*i].set_xlabel("Epsilon")
    axes[2*i].set_ylabel("Top-1 Accuracy")
    axes[2*i].set_title(title)
    axes[2*i].grid()

    # TOP3
    # SAE INSIGHTS shuffled DP
    color = "orchid"
    axes[2*i+1].plot(sae_insights_shuffled_dp_df["epsilon"], sae_insights_shuffled_dp_df["top3_acc"], 
                marker='o', color=color)
    lower_bound = sae_insights_shuffled_dp_df["top3_acc"] - sae_insights_shuffled_dp_df["top3_acc_std"]
    upper_bound = sae_insights_shuffled_dp_df["top3_acc"] + sae_insights_shuffled_dp_df["top3_acc_std"]
    axes[2*i+1].fill_between(sae_insights_shuffled_dp_df["epsilon"], lower_bound, upper_bound, color=color, alpha=0.15)
    

    # SAE INSIGHTS central DP
    color= "purple"
    axes[2*i+1].plot(sae_insights_df_central_dp["epsilon"], sae_insights_df_central_dp["top3_acc"], 
                marker='o', color=color)
    lower_bound = sae_insights_df_central_dp["top3_acc"] - sae_insights_df_central_dp["top3_acc_std"]
    upper_bound = sae_insights_df_central_dp["top3_acc"] + sae_insights_df_central_dp["top3_acc_std"]
    axes[2*i+1].fill_between(sae_insights_df_central_dp["epsilon"], lower_bound, upper_bound, color=color, alpha=0.15)

    axes[2*i+1].axhline(sae_insights_non_private_df["top3_acc"].item(), linestyle='--', color=color)


    # DENSE REPRESENTATION shuffled DP
    color = "lightseagreen"
    axes[2*i+1].plot(dense_representation_shuffled_dp_df["epsilon"], dense_representation_shuffled_dp_df["top3_acc"], 
                marker='o', color=color)
    lower_bound = dense_representation_shuffled_dp_df["top3_acc"] - dense_representation_shuffled_dp_df["top3_acc_std"]
    upper_bound = dense_representation_shuffled_dp_df["top3_acc"] + dense_representation_shuffled_dp_df["top3_acc_std"]
    axes[2*i+1].fill_between(dense_representation_shuffled_dp_df["epsilon"], lower_bound, upper_bound, color=color, alpha=0.15)

    # DENSE REPRESENTATION central DP
    color = "royalblue"
    axes[2*i+1].plot(dense_representation_df_central_dp["epsilon"], dense_representation_df_central_dp["top3_acc"], 
                marker='o', color=color)
    lower_bound = dense_representation_df_central_dp["top3_acc"] - dense_representation_df_central_dp["top3_acc_std"]
    upper_bound = dense_representation_df_central_dp["top3_acc"] + dense_representation_df_central_dp["top3_acc_std"]
    axes[2*i+1].fill_between(dense_representation_df_central_dp["epsilon"], lower_bound, upper_bound, color=color, alpha=0.15)

    axes[2*i+1].axhline(dense_representation_non_private_df["top3_acc"].item(), linestyle='--', color=color)

    axes[2*i+1].set_xlabel("Epsilon")
    axes[2*i+1].set_ylabel("Top-3 Accuracy")
    axes[2*i+1].set_title(title)
    axes[2*i+1].grid()
fig.suptitle("Shuffled and Central DP comparison")
fig.legend(handles=[line_sae_shuffled_DP, line_sae_central, line_dense_shuffled_DP, line_dense_central], 
           loc="lower center", 
           ncols=4,
           bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.005, 1, 1))
# %%
# Shuffled DP and central DP comparison:
