#%%
from classifier import LinearClassifier
from training import train_classifier, train_sae_supervised_data, train_sae_unsupervised_data, create_words_to_latents_table
from sae import VanillaSAE, TopKSAE, BatchTopKSAE, JumpReLUSAE
from activation_store import ActivationsStore
from basic_activation_store import BasicActivationsStore
from config import get_default_sae_cfg, post_init_sae_cfg, get_classifier_cfg, post_init_classifier_cfg
from transformer_lens import HookedTransformer
import torch
import json



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
        activations_store = BasicActivationsStore(model, cfg)
        train_sae_supervised_data(sae, activations_store, model, sae_cfg)

    activations_store_classifier = BasicActivationsStore(model, cfg)

    classifier = LinearClassifier(cfg["input_size"], cfg["num_classes"]).to(cfg["dtype"]).to(cfg["device"])

    train_classifier(sae, classifier, activations_store_classifier, cfg)
    
if __name__ == "__main__":
    sae_name = "gpt2-small_openwebtext_24576_topk_32_ft_dbpedia_14"
    with open(f"checkpoints/{sae_name}/config.json", "r") as f:
        sae_cfg = json.load(f)
    sae_cfg["dtype"] = torch.float32
    sae = sae_switch(sae_cfg["sae_type"], sae_cfg)
    sae.load_state_dict(torch.load(f"checkpoints/{sae_name}/sae.pt", weights_only=True))
    model = HookedTransformer.from_pretrained(sae_cfg["model_name"]).to(sae_cfg["dtype"]).to(sae_cfg["device"])
    table = create_words_to_latents_table(model, sae)
    table.to_csv(f"checkpoints/{sae_name}/words_to_latents.csv")