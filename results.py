# %%
import wandb
import pandas as pd
from main import wandb_project
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# %%
api = wandb.Api()
# projects = [wandb_project]
projects = ['our_butterfly_v5']

entity = "gorodissky-tel-aviv-university"
data = []
for project in projects: 

    runs = api.runs(f"{entity}/{project}")

    for run in runs:
        config_sae_type = run.config.get("sae_type", None)
        dict_size = run.config.get("dict_size", None)
        k = run.config.get("top_k", None)
        normalize_decoder_weights = run.config.get("normalize_decoder_weights_and_grad", None)
        butterfly_type = run.config.get("butterfly_type", None)

        final_l0_norm = run.summary.get("l0_norm", None)
        final_l2_loss = run.summary.get("l2_loss", None)
        final_ce_degradation = run.summary.get("performance/ce_degradation", None)
        final_num_dead_features = run.summary.get("num_dead_features", None)
        final_l1_norm = run.summary.get('l1_norm', None)
        runtime = run.summary.get('_runtime', None)
        runtime /= 60.
        l1_coeff = run.summary.get("l1_coeff", None)
        final_aux_loss = run.summary.get("aux_loss", None)
        
        
        data.append({
            "config_sae_type": config_sae_type,
            "dictionary_size": dict_size,
            "k": k,
            "l0_norm": final_l0_norm,
            "normalized_mse": final_l2_loss,
            "ce_degradation": final_ce_degradation,
            "num_dead_features": final_num_dead_features,
            "l1_norm": final_l1_norm,
            'runtime_mins': runtime,
            "l1_coeff": l1_coeff,
            "project": project,
            "aux_loss": final_aux_loss,
            "normlized_decoder_weights": normalize_decoder_weights,
            "butterfly_type": butterfly_type
        })

df = pd.DataFrame(data)


# %%
# Regular metrics
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Normalized MSE vs Dictionary size (k=32)
for sae_type in [ 'our_butterfly_topk', 'topk']:
    data = df[
        (df['config_sae_type'] == sae_type) &
        (df['k'] == 32.)
        ]
    data = data.sort_values(by='dictionary_size')
    axs[0, 0].plot(data['dictionary_size'], data['normalized_mse'], 
                   marker='o', linestyle='--', label=f"{sae_type} (k=32)")

axs[0, 0].set_title('Normalized MSE vs Dictionary size (k=32)')
axs[0, 0].set_xlabel('Dictionary size')
axs[0, 0].set_ylabel('Normalized MSE')
axs[0, 0].set_xscale('log')
axs[0, 0].legend()
axs[0, 0].grid(True)

# Plot 2: Normalized MSE vs k (Dict size = 12288)
for sae_type in ['our_butterfly_topk', 'topk']:
    data = df[
        (df['config_sae_type'] == sae_type) &
        (df['dictionary_size'] == 12288)
            ]
    data = data.sort_values(by='l0_norm')
    # label_suffix = '_standford' if 'standford' == butterfly_type else '_technion'
    axs[0, 1].plot(data['l0_norm'], data['normalized_mse'], 
                marker='o', linestyle='--', label=sae_type)

axs[0, 1].set_title('Normalized MSE vs k (Dict size = 12288)')
axs[0, 1].set_xlabel('k')
axs[0, 1].set_ylabel('Normalized MSE')
axs[0, 1].legend()
axs[0, 1].grid(True)

# Plot 3: CE degradation vs Dictionary size (k=32)
for sae_type in ['our_butterfly_topk', 'topk']:
    data = df[
            (df['config_sae_type'] == sae_type) &
            (df['k'] == 32)
            ]
    data = data.sort_values(by='dictionary_size')
    axs[1, 0].plot(data['dictionary_size'], data['ce_degradation'], 
                   marker='o', linestyle='--', label=f"{sae_type} (k=32)")

axs[1, 0].set_title('CE degradation vs Dictionary size (k=32)')
axs[1, 0].set_xlabel('Dictionary size')
axs[1, 0].set_ylabel('CE degradation')
axs[1, 0].set_xscale('log')
axs[1, 0].legend()
axs[1, 0].grid(True)


# Plot 4: CE degradation vs k (Dict size = 12288)
for sae_type in ['our_butterfly_topk', 'topk']:
    data = df[
        (df['config_sae_type'] == sae_type) &
        (df['dictionary_size'] == 12288)
            ]
    data = data.sort_values(by='l0_norm')
    # label_suffix = '_standford' if 'standford' == butterfly_type else '_technion'
    axs[1, 1].plot(data['l0_norm'], data['ce_degradation'], 
                marker='o', linestyle='--', label=sae_type)

axs[1, 1].set_title('CE degradation vs k (Dict size = 12288)')
axs[1, 1].set_xlabel('k')
axs[1, 1].set_ylabel('CE degradation')
axs[1, 1].legend()
axs[1, 1].grid(True)

# Adjust layout and display the plot
plt.tight_layout()






# %%
# runtime vs dictionary_size (k == 32)
fig, axes = plt.subplots()
for sae_type in ['our_butterfly_topk', 'topk']:
    data = df[
        (df['config_sae_type'] == sae_type) &
        (df['k'] == 32)
            ]
    data = data.sort_values(by='dictionary_size')
    # label_suffix = '_standford' if 'standford' == butterfly_type else '_technion'
    axes.plot(data['dictionary_size'], data['runtime_mins'], 
                marker='o', linestyle='--', label=sae_type)
axes.set_title('runtime vs dictionary size (k = 32)')
axes.set_xlabel('dictionary size')
# axes.set_xscale('log')
axes.set_ylabel('runtime (mins)')
axes.legend()
axes.grid(True)

# %%
# final_num_dead_features vs k (dict_size == 12288)
fig, axes = plt.subplots()
for sae_type in ['our_butterfly_topk', 'topk']:
    data = df[
        (df['config_sae_type'] == sae_type) &
        (df['dictionary_size'] == 12288)
                ]
    data = data.sort_values(by='l0_norm')
    # label_suffix = '_standford' if 'standford' == butterfly_type else '_technion'
    axes.plot(data['l0_norm'], data['num_dead_features'], 
                marker='o', linestyle='--', label=sae_type)
axes.set_title('Num dead features vs k (dictionary size = 12288)')
axes.set_xlabel('k')
axes.set_ylabel('final_num_dead_features')
axes.legend()
axes.grid(True)

# %%
# final auxillary loss vs k (dict_size == 12288)
fig, axes = plt.subplots()
for sae_type in ['our_butterfly_topk', 'topk']:
    data = df[
        (df['config_sae_type'] == sae_type) &
        (df['dictionary_size'] == 12288)
                ]
    data = data.sort_values(by='l0_norm')
    # label_suffix = '_standford' if 'standford' == butterfly_type else '_technion'
    axes.plot(data['l0_norm'], data['aux_loss'], 
                marker='o', linestyle='--', label=sae_type)
axes.set_title('Auxilary loss vs k (dictionary size = 12288)')
axes.set_xlabel('k')
axes.set_ylabel('final_aux_loss')
axes.legend()
axes.grid(True)
# %%
