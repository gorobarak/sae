from training import train_sae, train_sae_group
from sae import VanillaSAE, TopKSAE, BatchTopKSAE, JumpReLUSAE, ButterflySAE, ButterflyTopKSAE
from activation_store import ActivationsStore
from config import get_default_cfg, post_init_cfg
from transformer_lens import HookedTransformer


wandb_project = 'testing_top_k_butterfly'

# MSE vs dictionary size (k=32)
for sae_type in ["butterfly_topk", "topk"]:
    for expansion_factor in [8, 16, 32, 64]:
        cfg = get_default_cfg()
        
        # Model and Data
        cfg["model_name"] = "gpt2-small"
        cfg["dataset_path"] = "Skylion007/openwebtext"
        cfg['act_size'] = 768

        # Hook point
        cfg["layer"] = 8
        cfg["site"] = "resid_pre"
        
        # Optimization
        cfg["lr"] = 3e-4
        cfg["input_unit_norm"] = True
        cfg['device'] = 'cuda'

        # General 
        cfg["dict_size"] = 768 * expansion_factor
        cfg['l1_coeff'] = 0.
        cfg["sae_type"] = sae_type
        
        # WandB
        cfg['wandb_project'] = wandb_project
        
        # JumpRelu Specific
        # cfg['bandwidth'] = 0.001

        # TopK Specific
        cfg["top_k"] = 32
        cfg["aux_penalty"] = (1/32)
        

        
        if cfg["sae_type"] == 'butterfly':
            sae = ButterflySAE(cfg)
        elif cfg['sae_type'] == 'butterfly_topk':
            sae = ButterflyTopKSAE(cfg)
        elif cfg["sae_type"] == 'topk':
            sae = TopKSAE(cfg)


        cfg = post_init_cfg(cfg)
                
        model = HookedTransformer.from_pretrained(cfg["model_name"]).to(cfg["dtype"]).to(cfg["device"])
        activations_store = ActivationsStore(model, cfg)
        train_sae(sae, activations_store, model, cfg)

# MSE vs l0_norm  (dict_size =  16 * 768)
for sae_type in ["butterfly_topk",  "topk"]:
    # Don't retrain top_k=32
    for top_k in [16, 64, 128]:

        cfg = get_default_cfg()
        
        # Model and Data
        cfg["model_name"] = "gpt2-small"
        cfg["dataset_path"] = "Skylion007/openwebtext"
        cfg['act_size'] = 768

        # Hook point
        cfg["layer"] = 8
        cfg["site"] = "resid_pre"
        
        # Optimization
        cfg["lr"] = 3e-4
        cfg["input_unit_norm"] = True
        cfg['device'] = 'cuda'

        # General 
        cfg["dict_size"] = 768 * 16
        cfg['l1_coeff'] = 0.
        cfg["sae_type"] = sae_type

        # WandB
        cfg['wandb_project'] = wandb_project
        
        # JumpRelu Specific
        # cfg['bandwidth'] = 0.001

        # TopK Specific
        cfg["top_k"] = top_k
        cfg["aux_penalty"] = (1/32)
        
        
        if cfg["sae_type"] == 'butterfly':
            sae = ButterflySAE(cfg)
        elif cfg['sae_type'] == 'butterfly_topk':
            sae = ButterflyTopKSAE(cfg)
        elif cfg["sae_type"] == 'topk':
            sae = TopKSAE(cfg)


        cfg = post_init_cfg(cfg)
                
        model = HookedTransformer.from_pretrained(cfg["model_name"]).to(cfg["dtype"]).to(cfg["device"])
        activations_store = ActivationsStore(model, cfg)
        train_sae(sae, activations_store, model, cfg)


# Butterfly -  MSE vs l0_norm (dict_size = 16*768)
for l1_coeff in [0.004, 0.0018, 0.0008]:
    cfg = get_default_cfg()
    
    # Model and Data
    cfg["model_name"] = "gpt2-small"
    cfg["dataset_path"] = "Skylion007/openwebtext"
    cfg['act_size'] = 768

    # Hook point
    cfg["layer"] = 8
    cfg["site"] = "resid_pre"
    
    # Optimization
    cfg["lr"] = 3e-4
    cfg["input_unit_norm"] = True
    cfg['device'] = 'cuda'

    # General 
    cfg["dict_size"] = 768 * 16
    cfg['l1_coeff'] = l1_coeff
    cfg["sae_type"] = 'butterfly'
   
    # WandB
    cfg['wandb_project'] = wandb_project
    
    # JumpRelu Specific
    # cfg['bandwidth'] = 0.001

    # TopK Specific
    # cfg["top_k"] = 32
    # cfg["aux_penalty"] = (1/32)

    if cfg["sae_type"] == 'butterfly':
        sae = ButterflySAE(cfg)
    elif cfg['sae_type'] == 'butterfly_topk':
        sae = ButterflyTopKSAE(cfg)
    elif cfg["sae_type"] == 'topk':
        sae = TopKSAE(cfg)


    cfg = post_init_cfg(cfg)
            
    model = HookedTransformer.from_pretrained(cfg["model_name"]).to(cfg["dtype"]).to(cfg["device"])
    activations_store = ActivationsStore(model, cfg)
    train_sae(sae, activations_store, model, cfg)


