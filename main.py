from training import train_sae, train_sae_group
from sae import VanillaSAE, TopKSAE, BatchTopKSAE, JumpReLUSAE, ButterflySAE
from activation_store import ActivationsStore
from config import get_default_cfg, post_init_cfg
from transformer_lens import HookedTransformer


wandb_project = 'butterfly_compersion_v5'

# # MSE vs dictionary size (k=32)
# for sae_type in ["butterfly", "topk"]:
#     for expansion_factor in [4 ,8, 16, 32, 64]:
#         cfg = get_default_cfg()
#         cfg["model_name"] = "gpt2-small"
#         cfg["layer"] = 8
#         cfg["site"] = "resid_pre"
#         cfg["dataset_path"] = "Skylion007/openwebtext"
#         cfg["aux_penalty"] = (1/32)
#         cfg["lr"] = 3e-4
#         cfg["input_unit_norm"] = True
#         cfg["dict_size"] = 768 * expansion_factor
#         cfg['wandb_project'] = wandb_project
#         cfg['l1_coeff'] = 0.
#         cfg['act_size'] = 768
#         cfg['device'] = 'cuda'
#         cfg['bandwidth'] = 0.001
#         cfg["top_k"] = 32
#         cfg["sae_type"] = sae_type
        
#         if cfg["sae_type"] == 'butterfly':
#             sae = ButterflySAE(cfg)
#         elif cfg["sae_type"] == 'topk':
#             sae = TopKSAE(cfg)


#         cfg = post_init_cfg(cfg)
                
#         model = HookedTransformer.from_pretrained(cfg["model_name"]).to(cfg["dtype"]).to(cfg["device"])
#         activations_store = ActivationsStore(model, cfg)
#         train_sae(sae, activations_store, model, cfg)

# # MSE vs top_k (dict_size =  16 * 768)
# for sae_type in ["butterfly", "topk"]:
#     # Don't retrain top_k=32
#     for top_k in [16, 64, 128, 256]:
#         cfg = get_default_cfg()
#         cfg["model_name"] = "gpt2-small"
#         cfg["layer"] = 8
#         cfg["site"] = "resid_pre"
#         cfg["dataset_path"] = "Skylion007/openwebtext"
#         cfg["aux_penalty"] = (1/32)
#         cfg["lr"] = 3e-4
#         cfg["input_unit_norm"] = True
#         cfg["dict_size"] = 768 * 16
#         cfg['wandb_project'] = wandb_project
#         cfg['l1_coeff'] = 0.
#         cfg['act_size'] = 768
#         cfg['device'] = 'cuda'
#         cfg['bandwidth'] = 0.001
#         cfg["top_k"] = top_k
#         cfg["sae_type"] = sae_type
        
#         if cfg["sae_type"] == 'butterfly':
#             sae = ButterflySAE(cfg)
#         elif cfg["sae_type"] == 'topk':
#             sae = TopKSAE(cfg)


#         cfg = post_init_cfg(cfg)
                
#         model = HookedTransformer.from_pretrained(cfg["model_name"]).to(cfg["dtype"]).to(cfg["device"])
#         activations_store = ActivationsStore(model, cfg)
#         train_sae(sae, activations_store, model, cfg)


# Butterfly: MSE vs k
for l1_coeff in [0.004, 0.0018, 0.0008]:
    cfg = get_default_cfg()
    cfg["model_name"] = "gpt2-small"
    cfg["layer"] = 8
    cfg["site"] = "resid_pre"
    cfg["dataset_path"] = "Skylion007/openwebtext"
    cfg["aux_penalty"] = (1/32)
    cfg["lr"] = 3e-4
    cfg["input_unit_norm"] = True
    cfg["dict_size"] = 768 * 16
    cfg['wandb_project'] = wandb_project
    cfg['l1_coeff'] = l1_coeff
    cfg['act_size'] = 768
    cfg['device'] = 'cuda'
    cfg["sae_type"] = 'butterfly'
    cfg['use_top_k'] = False

    sae = ButterflySAE(cfg)


    cfg = post_init_cfg(cfg)
            
    model = HookedTransformer.from_pretrained(cfg["model_name"]).to(cfg["dtype"]).to(cfg["device"])
    activations_store = ActivationsStore(model, cfg)
    train_sae(sae, activations_store, model, cfg)