import transformer_lens.utils as utils
import torch 

def get_default_cfg():
    default_cfg = {
        # Optimization
        "seed": 49,
        "batch_size": 1024,
        "lr": 3e-4,
        "num_tokens": int(1e7), #total number of token trained on
        "beta1": 0.9,
        "beta2": 0.99,
        "max_grad_norm": 100000,
        "input_unit_norm": True,
        "device": "cuda",
        
        # Model and dataset parameters
        "seq_len": 128,
        "dtype": torch.float32,
        "model_name": "gpt2-small",
        "site": "resid_pre",
        "layer": 8,
        "act_size": 768,
        "dict_size": 12288,
        "model_batch_size": 512,
        "num_batches_in_buffer": 10,
        "dataset_path": "Skylion007/openwebtext",
        
        # WandB
        "wandb_project": "sparse_autoencoders",
        
        # Logging 
        "perf_log_freq": 100,
        
        "checkpoint_freq": 1000,
        

        # All 
        "sae_type": "topk",
        "n_batches_to_dead": 5,
        "normalize_decoder_weights_and_grad": False,
        
        # (Batch)TopKSAE 
        "top_k": 32,
        "top_k_aux": 512, #num of top k dead latens to use for aux loss
        "aux_penalty": (1/32), #auxilery lost coeffcient
        
        # JumpReLU
        "l1_coeff": 0,
        "bandwidth": 0.001,

        # Butterfly
        "butterfly_type": "standford"


    }
    default_cfg = post_init_cfg(default_cfg)
    return default_cfg

def post_init_cfg(cfg):
    cfg["hook_point"] = utils.get_act_name(cfg["site"], cfg["layer"])
    cfg["name"] = f"{cfg['model_name']}_{cfg['hook_point']}_{cfg['dict_size']}_{cfg['sae_type']}_{cfg['top_k'] if 'topk' in cfg['sae_type'] else cfg['l1_coeff']}_{cfg['lr']}"
    return cfg