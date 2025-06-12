import transformer_lens.utils as utils
import torch 

def get_default_sae_cfg():
    default_cfg = {
        "seed": 49,

        # Training 
        "batch_size": 10,
        "num_tokens": int(1e9),
        "lr": 3e-4,
        "beta1": 0.9,
        "beta2": 0.99,
        "max_grad_norm": 100000,
        "dtype": torch.float32,
        
        

        # Model 
        "model_name": "gpt2-small",
        "site": "resid_pre",
        "hook_point_layer": 8,
        "act_size": 768,
        "context_size" : 128,
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

    }
    default_cfg = post_init_sae_cfg(default_cfg)
    return default_cfg

def post_init_sae_cfg(cfg):
    cfg["dataset_name"] = cfg["dataset_path"].split("/")[-1]
    cfg["hook_point"] = utils.get_act_name(cfg["site"], cfg["hook_point_layer"])
    cfg["name"] = f"{cfg['model_name']}_{cfg['dataset_name']}_{cfg['dict_size']}_{cfg['sae_type']}_{cfg['top_k']}"
    return cfg

def get_default_classifier_cfg(sae_cfg):
    cfg = {
        "aggregate_function": "mean",
        "fine_tune": False,
        "baseline": False,
        "input_size": 768,
        "num_classes": 2,
        
        "dataset_path": "fancyzhx/dbpedia_14",
        "wandb_project": "classifiers",
        
        "lr": 1e-4,
        "beta1": 0.9,
        "beta2": 0.99,
        "num_samples_in_batch": 16,
        "num_samples_in_testset": 100,
        "filter_labels": [0, 2],
        "seed": 49,
        "device": "cuda",
        "dtype": torch.float32,
        "hook_point_layer": 8,
        "site": "resid_pre",
        "act_size": sae_cfg["act_size"],
        "model_name": sae_cfg["model_name"],
        "sae": sae_cfg["name"],
        

    }
    cfg = post_init_classifier_cfg(cfg, sae_cfg)
    return cfg

def post_init_classifier_cfg(cfg, sae_cfg):
    cfg["dataset_name"] = cfg["dataset_path"].split("/")[-1]
    cfg["input_size"] = sae_cfg["act_size"] if cfg["baseline"] else sae_cfg["dict_size"]
    cfg["hook_point"] = utils.get_act_name(cfg["site"], cfg["hook_point_layer"])

    cfg["name"] = f"classifier_{"baseline" if cfg["baseline"] else "X"}_{"ft" if cfg["fine_tune"] else "X"}_{cfg["aggregate_function"]}_{cfg["num_classes"]}_{cfg["sae"]}"
    return cfg