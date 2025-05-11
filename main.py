#%%
from classifier import LinearClassifier
from training import train_classifier, train_sae_supervised_data, train_sae_unsupervised_data
from sae import VanillaSAE, TopKSAE, BatchTopKSAE, JumpReLUSAE
from activation_store import ActivationsStore
from basic_activation_store import BasicActivationsStore
from config import get_default_sae_cfg, post_init_sae_cfg, get_classifier_cfg, post_init_classifier_cfg
from transformer_lens import HookedTransformer
import torch



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
        sae_cfg["dataset_name"] = cfg["dataset_name"]
        sae_cfg["name"] = sae_cfg["name"] + f"_ft_{cfg['dataset_name']}"
        activations_store = BasicActivationsStore(model, cfg)
        train_sae_supervised_data(sae, activations_store, model, sae_cfg)

    activations_store_classifier = BasicActivationsStore(model, cfg)

    classifier = LinearClassifier(cfg["input_size"], cfg["num_classes"]).to(cfg["dtype"]).to(cfg["device"])

    train_classifier(sae, classifier, activations_store_classifier, cfg)


if __name__ == "__main__":

    sae_cfg = get_default_sae_cfg()
    # SAE
    sae_cfg["sae_type"] = "topk"
    sae_cfg["input_unit_norm"] = True
    sae_cfg["dict_size"] = 768 * 32
    

    # Model
    sae_cfg["model_name"] = "gpt2-small"
    sae_cfg["layer"] = 8
    sae_cfg["site"] = "resid_pre"
    sae_cfg['act_size'] = 768

    # Dataset
    sae_cfg["dataset_path"] = "Skylion007/openwebtext"

    # Training
    sae_cfg["lr"] = 3e-4
    sae_cfg['device'] = 'cuda'
    sae_cfg['num_tokens'] = int(1e7)
    sae_cfg['batch_size'] = 1024
    
    # Logging
    sae_cfg['wandb_project'] = 'pretraining_saes'
    
    
    # TopKSAE specific
    sae_cfg["top_k"] = 32
    sae_cfg["aux_penalty"] = (1/32)
    
    # JumpReLU specific
    sae_cfg['bandwidth'] = 0.001
    sae_cfg['l1_coeff'] = 0.0018

    sae_cfg = post_init_sae_cfg(sae_cfg)


    #path = pretrain_sae(sae_cfg)
    path = f"./checkpoints/{sae_cfg['name']}/sae.pt"

    classifier_cfg = get_classifier_cfg(sae_cfg)
    
    classifier_cfg["wandb_project"] = "final_with_accuracy"
    classifier_cfg["aggregate_function"] = "max"
   
    # begin{basline = False}
    classifier_cfg["basline"] = False    
    
    classifier_cfg["fine_tune"] = False
    classifier_cfg = post_init_classifier_cfg(classifier_cfg, sae_cfg)
    train_classifier_sae(sae_cfg, classifier_cfg, path_to_pt_sae=path)

 
    classifier_cfg["fine_tune"] = True
    classifier_cfg = post_init_classifier_cfg(classifier_cfg, sae_cfg)
    train_classifier_sae(sae_cfg, classifier_cfg, path_to_pt_sae=path)

    # end{baseline = False}

    # begin{baseline = True}
    # In baseline there is no effect for fintuning as it trained directly on residual stream activations
    classifier_cfg["baseline"] = True
    
    classifier_cfg["fine_tune"] = False
    classifier_cfg = post_init_classifier_cfg(classifier_cfg, sae_cfg)
    train_classifier_sae(sae_cfg, classifier_cfg, path_to_pt_sae=path)
    # end{baseline = True}

    
