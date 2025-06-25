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
        "dict_size": 768*32,
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
        "reactivation": False,
        "num_classes": 14,
        
        "dataset_path": "fancyzhx/dbpedia_14",
        "wandb_project": "classifiers",
        
        "lr": 5e-5,
        "beta1": 0.9,
        "beta2": 0.99,
        "num_samples_in_batch": 512,
        "filter_labels": [0, 2],
        "seed": 49,
        "device": "cuda",
        "dtype": torch.float32,
        "hook_point_layer": sae_cfg["hook_point_layer"],
        "hook_point": sae_cfg["hook_point"],
        "act_size": sae_cfg["d_in"],
        "model_name": sae_cfg["model_name"],
        "sae": sae_cfg["neuronpedia_id"],
        "log_acc_freq" : 100,
        

    }
    cfg = post_init_classifier_cfg(cfg, sae_cfg)
    return cfg

def post_init_classifier_cfg(cfg, sae_cfg):
    cfg["dataset_name"] = cfg["dataset_path"].split("/")[-1]
    cfg["input_size"] = sae_cfg["d_in"] if cfg["baseline"] else sae_cfg["d_sae"]
    # cfg["hook_point"] = utils.get_act_name(cfg["site"], cfg["hook_point_layer"])
    cfg["name"] = f"classifier_{"reactivation" if cfg["reactivation"] else "X"}_{cfg["aggregate_function"]}_{cfg["num_classes"]}_{cfg["sae"]}"
    return cfg