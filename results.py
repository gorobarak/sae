# %%
import wandb
import pandas as pd

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


# %%
sae_name = "gpt2-small_openwebtext_24576_topk_32"
df = pd.read_csv(f"checkpoints/{sae_name}/words_to_latents.csv")
df
# %%
df.describe()
# %%
col0 = df.iloc[:, 0]
df = df.drop(columns=[col0.name])
df
# %%
import transformer_lens
import torch
# %%
gpt2 = transformer_lens.HookedTransformer.from_pretrained("gpt2-small")

# %%
columns = df.columns.map(lambda x: torch.tensor([x]))
columns = columns.map(gpt2.to_str_tokens)
columns = columns.map(lambda x: x[0] if type(x) is list else x)
df.columns = columns
df
# %%
df.columns
# %%
df
# %%
def mean_over_nonzeros(col):
    non_zeros = col[col != 0]
    if len(non_zeros) == 0:
        return 0
    else:
        return non_zeros.mean()

# %%
means = df.apply(mean_over_nonzeros, axis=0)

# %%
means.plot(kind='bar', figsize=(12, 6))
plt.xlabel('Token')
plt.ylabel('Mean Activation')
# %%

# %%
# Top 5 activating tokens for each latent
def top_activating_tokens(row, k=5):
    top_tokens_vals = row.nlargest(k)
    top_tokens = top_tokens_vals.index
    return top_tokens, top_tokens_vals
# %%
res = df.apply(top_activating_tokens, axis=1)
res
# %%
res.iloc[8196]
# %%
mini_df = df.iloc[:, 0:10]
# %%

# %%
# gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# %%
df.columns = range(df.shape[1])

# %%
res = gpt2_tokenizer.convert_ids_to_tokens([50256])
res
# def token_id_to_string_rep(token_id)


# %%
res = mini_df.columns.map(gpt2_tokenizer.convert_ids_to_tokens)
# %%
mini_df.columns
# %%
res
# %%
mini_df.columns = res
# %%
mini_df


# %%
def token_id_to_string_rep(token_id):
    token = gpt2_tokenizer.convert_ids_to_tokens(token_id)
    if type(token) is list:
        token = token[0]
    return token
# %%
col0 = df.iloc[:, 0]
col1000 = df.iloc[:, 1000]
col50000 = df.iloc[:, 50000]
nonzero_col0 = col0[col0 != 0]
nonzero_col1000 = col1000[col1000 != 0]
nonzero_col50000 = col50000[col50000 != 0]
print(nonzero_col0.sort_values(ascending=False))
#%%
print(nonzero_col1000.sort_values(ascending=False))
#%%
print(nonzero_col50000.sort_values(ascending=False))
# %%
diff1 = nonzero_col0 - nonzero_col1000
diff1.dropna()

# %%
diff2 = nonzero_col0 - nonzero_col50000
diff2.dropna()

# %%
