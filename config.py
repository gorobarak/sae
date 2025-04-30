import transformer_lens.utils as utils
import torch 

def get_default_cfg():
    default_cfg = {
        "seed": 49,

        # Training 
        "batch_size": 4096,
        "num_tokens": int(1e9),
        "lr": 3e-4,
        "beta1": 0.9,
        "beta2": 0.99,
        "max_grad_norm": 100000,
        "seq_len": 128,
        "dtype": torch.float32,

        # Model 
        "model_name": "gpt2-small",
        "site": "resid_pre",
        "layer": 8,
        "act_size": 768,
        "model_batch_size": 512,
        "num_batches_in_buffer": 10,

        # SAE 
        "sae_type": "topk",
        "dict_size": 12288,
        "device": "cuda",
        "input_unit_norm": True,
        "n_batches_to_dead": 5,
        
        # Dataset 
        "dataset_path": "Skylion007/openwebtext",
        
        # Logging
        "wandb_project": "sparse_autoencoders",
        "perf_log_freq": 1000,
        "checkpoint_freq": 10000,
        

        # (Batch)TopKSAE specific
        "top_k": 32,
        "top_k_aux": 512,
        "aux_penalty": (1/32),
        
        # JumpRelu specific
        "bandwidth": 0.001,
        "l1_coeff": 0,

        # Classification
        "aggregate_function": "mean",
        "num_samples_in_batch": 10,
    }
    default_cfg = post_init_cfg(default_cfg)
    return default_cfg

def post_init_cfg(cfg):
    cfg["hook_point"] = utils.get_act_name(cfg["site"], cfg["layer"])
    cfg["name"] = f"{cfg['model_name']}_{cfg['hook_point']}_{cfg['dict_size']}_{cfg['sae_type']}_{cfg['top_k']}_{cfg['lr']}"
    return cfg