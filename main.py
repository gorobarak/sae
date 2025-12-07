
import os
os.environ["HF_HOME"] = "/home/yandex/APDL2425a/group_12/gorodissky/.cache/huggingface"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
from sentence_transformers import SentenceTransformer
from  probes import create_dataset_baseline, create_datasets, code_words, math_words, tl_model_names
import torch
from transformer_lens import HookedTransformer
from transformer_lens.loading_from_pretrained import get_official_model_name
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


dataset_name = "allenai/WildChat-1M"
# pooling_strategies = ["last"]
dataset_size = 32 * 100
batch_size = 32

for tl_model_name in ["phi-3"]: #tl_model_names.values():
    hf_model_name = get_official_model_name(tl_model_name)
    model = AutoModelForCausalLM.from_pretrained(hf_model_name)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    token_ids = sum(tokenizer(math_words, add_special_tokens=False)["input_ids"], [])
    print(f"processing {hf_model_name}", file=sys.stderr)
    Xs, Ys = create_datasets(
        model,
        tl_model_name,
        tokenizer,
        dataset_name,
        dataset_size=dataset_size,
        batch_size=batch_size,
        tokens_ids=token_ids
    )
    path = f"data/pred_gen/{dataset_name}/{tl_model_name}"
    os.makedirs(path, exist_ok=True)
    for label_name, labels in Ys.items():
        torch.save(labels,f"{path}/Y_{label_name}.pt")

    for layer_idx, pooling_dict in Xs.items():
        for pooling_strategy, acts in pooling_dict.items():
            os.makedirs(f"{path}/layer_{layer_idx}", exist_ok=True)
            torch.save(acts, f"{path}/layer_{layer_idx}/X_{pooling_strategy}.pt")



