#%%
from classifier import LinearClassifier
from training import *
from sae import VanillaSAE, TopKSAE, BatchTopKSAE, JumpReLUSAE
from activation_store import ActivationsStore
from config import get_default_sae_cfg, post_init_sae_cfg, get_classifier_cfg, post_init_classifier_cfg
from transformer_lens import HookedTransformer
import torch
import json
from sae_lens import SAE
from my_activation_store import UnsupervisedActivationStore, SupervisedActivationsStore
from datasets import load_dataset



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
    
if __name__ == "__main__":
    release = "gpt2-small-res-jb"
    sae_id = "blocks.8.hook_resid_pre"
    sae, cfg, _ = SAE.from_pretrained(release, sae_id, device="cuda")
    cfg['device'] = 'cuda'
    cfg['dtype'] = torch.float32
    cfg['num_sequences'] = int(1e5)
    cfg["batch_size"] = 256
    cfg['ctx_size'] = 128
    
    # model = HookedTransformer.from_pretrained(cfg["model_name"]).to(cfg["dtype"]).to(cfg["device"])

    # print(model.to_tokens("Hello, world!"))

    # dataset = iter(load_dataset(cfg["dataset_path"], split="train", streaming=True))
    
    # get_top_activating_samples(model, sae, cfg, dataset, duplicate_tokens=False, k=10)
    generate_descriptions("checkpoints/topk_samples_duplicate/heaps_390.pkl", explainer_model="gpt-4o-mini")
