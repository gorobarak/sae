#%%
from classifier import LinearClassifier
from training import train_sae, train_sae_group, train_classifier
from sae import VanillaSAE, TopKSAE, BatchTopKSAE, JumpReLUSAE
from activation_store import ActivationsStore
from basic_activation_store import BasicActivationsStore
from config import get_default_cfg, post_init_cfg
from transformer_lens import HookedTransformer



def sae_switch(sae_type):
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

if __name__ == "__main__":
    cfg = get_default_cfg()
    # SAE
    cfg["sae_type"] = "topk"
    cfg["input_unit_norm"] = True
    cfg["dict_size"] = 768 * 32
    

    # Model
    cfg["model_name"] = "gpt2-small"
    cfg["layer"] = 8
    cfg["site"] = "resid_pre"
    cfg['act_size'] = 768

    # Dataset
    cfg["dataset_path"] = "fancyzhx/dbpedia_14"

    # Training
    cfg["lr"] = 3e-4
    cfg['device'] = 'cuda'
    cfg['num_samples_in_batch'] = 16

    # Logging
    cfg['wandb_project'] = 'sae_trained_on_dbpedia'
    
    
    
    # TopKSAE specific
    cfg["top_k"] = 32
    cfg["aux_penalty"] = (1/32)
    
    # JumpReLU specific
    cfg['bandwidth'] = 0.001
    cfg['l1_coeff'] = 0.0018


    sae = sae_switch(cfg["sae_type"])
    cfg = post_init_cfg(cfg)
                
    model = HookedTransformer.from_pretrained(cfg["model_name"]).to(cfg["dtype"]).to(cfg["device"])
    activations_store = BasicActivationsStore(model, cfg)
    train_sae(sae, activations_store, model, cfg)

    activations_store_classifier = BasicActivationsStore(model, cfg)

    num_classes = 2
    sae_classifier = LinearClassifier(cfg["dict_size"], num_classes).to(cfg["dtype"]).to(cfg["device"])
    train_classifier(sae, sae_classifier, activations_store_classifier, cfg)

