from training import train_sae, train_sae_group
from sae import VanillaSAE, TopKSAE, BatchTopKSAE, JumpReLUSAE, ButterflySAE
from activation_store import ActivationsStore
from config import get_default_cfg, post_init_cfg
from transformer_lens import HookedTransformer

for sae_type in ["topk, butterfly"]:
    cfg = get_default_cfg()
    cfg["model_name"] = "gpt2-small"
    cfg["layer"] = 8
    cfg["site"] = "resid_pre"
    cfg["dataset_path"] = "Skylion007/openwebtext"
    cfg["aux_penalty"] = (1/32)
    cfg["lr"] = 3e-4
    cfg["input_unit_norm"] = True
    cfg["dict_size"] = 768 * 16
    cfg['wandb_project'] = 'butterfly_compersion'
    cfg['l1_coeff'] = 0.
    cfg['act_size'] = 768
    cfg['device'] = 'cuda'
    cfg['bandwidth'] = 0.001
    cfg["top_k"] = 32
    cfg["sae_type"] = sae_type
    # sae = None 
    
    if sae_type == 'butterfly':
        sae = ButterflySAE(cfg)
    elif sae_type == 'topk':
        sae = TopKSAE(cfg)

    cfg = post_init_cfg(cfg)
            
    model = HookedTransformer.from_pretrained(cfg["model_name"]).to(cfg["dtype"]).to(cfg["device"])
    activations_store = ActivationsStore(model, cfg)
    train_sae(sae, activations_store, model, cfg)