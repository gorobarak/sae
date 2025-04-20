from training import train_sae
from sae import TopKSAE, BF_SAE, BF_TopKSAE, ButterflyTopKSAE, ButterflySAE, OurButterflyTopKSAE, ButterflyUnitaryTopKSAE
from activation_store import ActivationsStore
from config import get_default_cfg, post_init_cfg
from transformer_lens import HookedTransformer


# Checklist before running:
# 1. Check the wandb project name
# 2.
# 3. 
# 4. 

wandb_project = 'our_butterfly_v5'

def sae_type_switch(sae_type, cfg):
    sae = None
    match sae_type:
        case 'bf':
            sae = BF_SAE(cfg)
        case 'bf_topk':
            sae = BF_TopKSAE(cfg)
        case 'butterfly':
            sae = ButterflySAE(cfg)
        case 'butterfly_topk':
            sae = ButterflyTopKSAE(cfg)
        case 'butterfly_unitary_topk':
            sae = ButterflyUnitaryTopKSAE(cfg)
        case 'topk':
            sae = TopKSAE(cfg)
        case 'our_butterfly_topk':
            sae = OurButterflyTopKSAE(cfg)
        case _:
            raise ValueError(f"Unknown SAE type: {sae_type}")
    return sae

if __name__ == '__main__':

    # MSE vs dictionary size (k=32)
    for sae_type in ["our_butterfly_topk", "topk"]:
        for expansion_factor in [8 ,16, 20, 26]:
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
            
            # Butterfly 
            # cfg["butterfly_type"] = 'standford

            
            sae = sae_type_switch(cfg['sae_type'], cfg)


            cfg = post_init_cfg(cfg)
                    
            model = HookedTransformer.from_pretrained(cfg["model_name"]).to(cfg["dtype"]).to(cfg["device"])
            activations_store = ActivationsStore(model, cfg)
            train_sae(sae, activations_store, model, cfg)

    # MSE vs l0_norm  (dict_size =  16 * 768)
    for sae_type in ["our_butterfly_topk", "topk"]:
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
            cfg['normalize_decoder_weights_and_grad'] = False

            # WandB
            cfg['wandb_project'] = wandb_project
            
            # JumpRelu Specific
            # cfg['bandwidth'] = 0.001

            # TopK Specific
            cfg["top_k"] = top_k
            cfg["aux_penalty"] = (1/32)
            cfg["top_k_aux"] = 512 #num of top k dead latens to use for aux loss
            
            # Butterfly 
            # cfg["butterfly_type"] = 'standford'


            sae = sae_type_switch(cfg['sae_type'], cfg)


            cfg = post_init_cfg(cfg)
                    
            model = HookedTransformer.from_pretrained(cfg["model_name"]).to(cfg["dtype"]).to(cfg["device"])
            activations_store = ActivationsStore(model, cfg)
            train_sae(sae, activations_store, model, cfg)


    # # Baseline MSE vs l0_norm  (dict_size =  16 * 768)
    # for top_k in [16, 32, 64, 128]:

    #     cfg = get_default_cfg()
            
    #     # Model and Data
    #     cfg["model_name"] = "gpt2-small"
    #     cfg["dataset_path"] = "Skylion007/openwebtext"
    #     cfg['act_size'] = 768

    #     # Hook point
    #     cfg["layer"] = 8
    #     cfg["site"] = "resid_pre"
        
    #     # Optimization
    #     cfg["lr"] = 3e-4
    #     cfg["input_unit_norm"] = True
    #     cfg['device'] = 'cuda'

    #     # General 
    #     cfg["dict_size"] = 768 * 16 
    #     cfg['l1_coeff'] = 0.
    #     cfg["sae_type"] = "topk"
    #     cfg['normalize_decoder_weights_and_grad'] = False

    #     # WandB
    #     cfg['wandb_project'] = wandb_project
        
    #     # JumpRelu Specific
    #     # cfg['bandwidth'] = 0.001

    #     # TopK Specific
    #     cfg["top_k"] = top_k
    #     cfg["aux_penalty"] = (1/32)
    #     cfg["top_k_aux"] = 512 #num of top k dead latens to use for aux loss
            
    #     # Butterfly 
    #     cfg["butterfly_type"] = "standford"


    #     sae = sae_type_switch(cfg['sae_type'], cfg)


    #     cfg = post_init_cfg(cfg)
                
    #     model = HookedTransformer.from_pretrained(cfg["model_name"]).to(cfg["dtype"]).to(cfg["device"])
    #     activations_store = ActivationsStore(model, cfg)
    #     train_sae(sae, activations_store, model, cfg) 




