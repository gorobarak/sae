from matplotlib.pyplot import cla
from classifier import LinearClassifier
from training import *
from sae import VanillaSAE, TopKSAE, BatchTopKSAE, JumpReLUSAE
from activation_store import ActivationsStore
from config import get_default_sae_cfg, post_init_sae_cfg, get_default_classifier_cfg, post_init_classifier_cfg
from transformer_lens import HookedTransformer
import torch
import json
from sae_lens import SAE
from my_activation_store import UnsupervisedActivationStore, SupervisedActivationsStore
from datasets import load_dataset
import requests
import os



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
    print(message)
    lines.append("\n" + message)

if __name__ == "__main__":
    release = "gpt2-small-res-jb"
    sae_id = "blocks.9.hook_resid_pre"
    sae, sae_cfg, _ = SAE.from_pretrained(release, sae_id, device="cuda")

    for class_idx in range(14):
        dir_path = f"checkpoints/dbpedia_class_{class_idx}_gpt2-small_blocks.8.hook_resid_post"
        
        # load activatoins
        with open(os.path.join(dir_path, "activations.pkl"), "rb") as f:
            activations = torch.load(f).to('cuda') # [batch, seq, d_model]

        # aggregate activations along sequence and batch dimensions
        agg_activation = activations.mean(dim=(0, 1)) # [d_model]
        
        # get dictionary activation
        with torch.no_grad():
            dict_activations = sae.encode(agg_activation.unsqueeze(0)).squeeze(0) # [d_sae]
        
        # get top-k features
        dict_topk = torch.topk(dict_activations, k=15, sorted=True)

        # get descriptions for top-k features
        model_id = "gpt2-small"
        layer = "9-res-jb"
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
                for explanation in body["explanations"]:
                    print_and_write(f"Feature {feature_idx}", lines)
                    print_and_write(f"{act_val.item()} -- {explanation['description']}", lines)
                    print_and_write("---------------", lines)
            else:
                print_and_write(f"Failed to fetch explanations: {r.status_code} - {r.reason}", lines)
        
        # write to file
        with open(os.path.join(dir_path, "top_features.txt"), "w") as f:
            f.writelines(lines)
