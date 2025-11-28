
import os
os.environ["HF_HOME"] = "/home/yandex/APDL2425a/group_12/gorodissky/.cache/huggingface"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
from sentence_transformers import SentenceTransformer
from  probes import create_datasets, code_words
import torch
from transformer_lens import HookedTransformer
from transformer_lens.loading_from_pretrained import get_official_model_name
from transformers import AutoTokenizer
from datasets import load_dataset


dataset_name = "allenai/WildChat-1M"
tl_model_names = ["mistral-7b-instruct", "phi-3"]

for tl_model_name in tl_model_names:
    hf_model_name = get_official_model_name(tl_model_name)
    model = HookedTransformer.from_pretrained(tl_model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    dataset_size = int(1e3)
    batch_size = 16

    Xs, Ys = create_datasets(
        model,
        hf_model_name,
        tokenizer,
        dataset_name,
        dataset_size=dataset_size,
        batch_size=batch_size,
        record_nll_telemetry=False,
        record_tokens_mass_telemetry=False,
        record_length_telemetry=True,
    )

    for hookpoint, X in Xs.items():
        for task, Y in Ys.items():
            save_dir = f"data/{task}/{tl_model_name}/{hookpoint}"
            os.makedirs(save_dir, exist_ok=True)
            torch.save(Y, os.path.join(save_dir, "Y.pt"))
            torch.save(X, os.path.join(save_dir, "X.pt"))