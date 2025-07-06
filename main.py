from fastapi.exception_handlers import request_validation_exception_handler
from sympy import use
from classifier import LinearClassifier
from training import *
from sae import VanillaSAE, TopKSAE, BatchTopKSAE, JumpReLUSAE
from activation_store import ActivationsStore
from config import get_default_sae_cfg, post_init_sae_cfg, get_default_classifier_cfg, post_init_classifier_cfg
from transformer_lens import HookedTransformer
import torch
from sae_lens import SAE
from my_activation_store import UnsupervisedActivationStore, SupervisedActivationsStore
from datasets import load_dataset
import requests
import os
from query_gpt import query_gpt, build_descriminate_task_prompt
import sys
import random



def sae_switch(sae_type, cfg):
    match sae_type:
        case "vanilla":
            return VanillaSAE(cfg)
        case "topk":
            return TopKSAE(cfg)
        case "batchtopk":
            return BatchTopKSAE(cfg)
        case "jumprelu":
            return JumpReLUSAE(cfg)
        case _:
            raise ValueError(f"Unknown SAE type: {cfg['sae_type']}")
        
def pretrain_sae(cfg):

    sae = sae_switch(cfg["sae_type"], cfg)
    model = HookedTransformer.from_pretrained(cfg["model_name"]).to(cfg["dtype"]).to(cfg["device"])
    activations_store = ActivationsStore(model, cfg)
    train_sae_unsupervised_data(sae, activations_store, model, cfg)
    
    path = f"./checkpoints/{cfg['name']}/sae.pt"
    return path

def train_classifier_sae(sae_cfg, cfg, path_to_pt_sae=None):
   
    sae = sae_switch(sae_cfg["sae_type"], sae_cfg)
   
    sae.load_state_dict(torch.load(path_to_pt_sae))
    model = HookedTransformer.from_pretrained(cfg["model_name"]).to(cfg["dtype"]).to(cfg["device"])

    if cfg["fine_tune"]:
        sae_cfg["dataset_name"] = sae_cfg["dataset_name"] + "+" + cfg["dataset_name"]
        sae_cfg["name"] = sae_cfg["name"] + f"_ft_{cfg['dataset_name']}"
        activations_store = SupervisedDataActivationsStore(model, cfg)
        train_sae_supervised_data(sae, activations_store, model, sae_cfg)

    activations_store_classifier = SupervisedDataActivationsStore(model, cfg)

    classifier = LinearClassifier(cfg["input_size"], cfg["num_classes"]).to(cfg["dtype"]).to(cfg["device"])

    train_classifier(sae, classifier, activations_store_classifier, cfg)


def print_and_write(message, lines):
    print(message, file=sys.stderr)
    lines.append("\n" + message)

def generate_population_level_insights(num_dbpedia_classes=7, reactivations=False):
    
    model_name = "gemma-2-2b"
    hook_point = "blocks.24.hook_resid_post"
    sae_release = "gemma-scope-2b-pt-res-canonical"
    sae_id = "layer_24/width_16k/canonical" 
    sae = SAE.from_pretrained(sae_release, sae_id, device="cuda")[0]

    for class_idx in range(num_dbpedia_classes):
        dir_path = f"checkpoints/dbpedia_class_{class_idx}_{model_name}_{hook_point}"
        
        # load activatoins / reactivations
        if reactivations:
            with open(os.path.join(dir_path, "reactivations.pt"), "rb") as f:
                activations = torch.load(f) # [batch, seq, d_model] on CPU
        else:
             with open(os.path.join(dir_path, "activations.pt"), "rb") as f:
                activations = torch.load(f) # [batch, seq, d_model] on CPU

        # aggregate activations along sequence and batch dimensions
        agg_activation = activations.mean(dim=(0, 1)) # [d_model]
        
        
        
        # get dictionary activation
        agg_activation = agg_activation.to('cuda')  # Move to GPU before passing into SAE
        with torch.no_grad():
            dict_activations = sae.encode(agg_activation.unsqueeze(0)).squeeze(0) # [d_sae]
        
        # get top-k features
        dict_topk = torch.topk(dict_activations, k=15, sorted=True)

        # get descriptions for top-k features
        model_id = "gemma-2-2b"
        layer = "24-gemmascope-res-16k"
        lines = []
        print_and_write(f"Top features for class {class_idx}", lines)
        for feature_idx, act_val in zip(dict_topk.indices, dict_topk.values):
            if act_val == 0.0:
                break
            r = requests.get(
            f"https://www.neuronpedia.org/api/feature/{model_id}/{layer}/{feature_idx}"
            )
            if r.status_code == 200:
                body = r.json()
                explanation = body["explanations"][-1]
                print_and_write(f"Feature {feature_idx}", lines)
                print_and_write(f"{act_val.item()} -- {explanation['description']}", lines)
                print_and_write("---------------", lines)
            else:
                print_and_write(f"Failed to fetch explanations: {r.status_code} - {r.reason}", lines)
        
        # write to file
        filename = "reactivations_" if reactivations else "activations_"
        filename += "top_features.txt"
        with open(os.path.join(dir_path, filename), "w") as f:
            f.writelines(lines)

def descriminate_task(use_reactivations=False, use_same_class_as_decoy=False):
    real_class_idx = 0
    decoy_class_idx = real_class_idx if use_same_class_as_decoy else 2
    NUM_SAMPLES_IN_CLASS = 40000
    num_of_tests = 250
    offset_real = real_class_idx * NUM_SAMPLES_IN_CLASS
    offset_decoy = decoy_class_idx * NUM_SAMPLES_IN_CLASS
    offset_decoy += num_of_tests if use_same_class_as_decoy else 0
    model_name = "gpt2-small"
    hook_point = "blocks.8.hook_resid_pre"
    real_class_path = f"checkpoints/dbpedia_class_{real_class_idx}_{model_name}_{hook_point}"
    model = HookedTransformer.from_pretrained(model_name, device="cuda", dtype=torch.float32)
    tokenizer = model.tokenizer
    dataset = load_dataset("fancyzhx/dbpedia_14", split="train")
    sae_release= "gpt2-small-res-jb"
    sae_id = "blocks.8.hook_resid_pre" 
    sae, sae_cfg, _ = SAE.from_pretrained(sae_release, sae_id, device="cuda")
    
    # Load activations or reactivations
    if use_reactivations:
        with open(os.path.join(real_class_path, "reactivations.pt"), "rb") as f:
            real_activations = torch.load(f).to('cuda')

    else:
        with open(os.path.join(real_class_path, "activations.pt"), "rb") as f:
            real_activations = torch.load(f).to('cuda')
    
    correct = 0
    for i in range(num_of_tests):
        real_act = real_activations[i] # [seq_len, d_model]

        real_text = dataset[i + offset_real]['content']
        real_text = tokenizer.decode(model.to_tokens(real_text, prepend_bos=True, truncate=True).squeeze(0)[:128])

        decoy_text1 = dataset[i * 2 + offset_decoy]['content']
        decoy_text1 = tokenizer.decode(model.to_tokens(decoy_text1, prepend_bos=True, truncate=True).squeeze(0)[:128])

        decoy_text2 = dataset[(i * 2 + 1) + offset_decoy]['content']
        decoy_text2 = tokenizer.decode(model.to_tokens(decoy_text2, prepend_bos=True, truncate=True).squeeze(0)[:128])  
        
        dict_real_acts = sae.encode(real_act.unsqueeze(0)).squeeze(0)  # [seq_len, d_sae]
        
        agg_dict_acts = dict_real_acts.max(dim=0).values # [d_sae] / aggregate dict activations 

        top10_real = torch.topk(agg_dict_acts, k=10, sorted=True)

        real_idx = random.sample(range(3), 1)[0] + 1 # 1, 2, or 3
    
        prompt = build_descriminate_task_prompt(real_text, decoy_text1, decoy_text2, top10_real, real_idx, hook_point)

        response = query_gpt(prompt, model="gpt-4o-mini")
        response = response.strip().lower()
        if response.startswith(str(real_idx)):
            correct += 1
        elif response.startswith(str((real_idx + 1) % 3 + 1)) or response.startswith(str(real_idx % 3 + 1)):
            pass
        else:
            print(f"Error: {response} isn't a valid response",file=sys.stderr)

    accuracy = correct / num_of_tests
    filename = ("reactivations" if use_reactivations else "activations") + "_descriminate_task_results.txt"
    if use_same_class_as_decoy:
        filename = "same_class_decoy_" + filename
    with open(os.path.join(real_class_path, filename), "w") as f:
        f.write(f"Accuracy: {accuracy:.2f}\n")
        f.write(f"Correct: {correct}\n")
        f.write(f"Total tests: {num_of_tests}\n")

     

if __name__ == "__main__":
    
    # t = torch.randint(0, 100, (10,))
    # top10_real = torch.topk(t, k=10, sorted=True)
    # prompt = build_descriminate_task_prompt("real_text", "decoy_text1", "decoy_text2", top10_real, 1, "blocks.8.hook_resid_pre")
    # print(prompt)
    
    # descriminate_task(use_reactivations=False, use_same_class_as_decoy=True)

    generate_population_level_insights(num_dbpedia_classes=7, reactivations=True)