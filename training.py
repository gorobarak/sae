from sys import activate_stack_trampoline
import torch
import tqdm
from logs import init_wandb, log_wandb, log_model_performance, save_checkpoint
import pandas as pd


def create_words_to_latents_table(model, sae):
    vocab_size = model.cfg.d_vocab
    dict_size = sae.cfg["dict_size"]
    topk = sae.cfg["top_k"]
    table = pd.DataFrame(0, index=range(dict_size), columns=range(vocab_size), dtype=float)
    for token_id in range(vocab_size):
        token_id_tensor = torch.tensor([[token_id]], device=sae.cfg["device"])
        with torch.no_grad():
            _, cache = model.run_with_cache(
                token_id_tensor,
                names_filter=[sae.cfg["hook_point"]],
                stop_at_layer=sae.cfg["layer"] + 1,
            )
            activation = cache[sae.cfg["hook_point"]]
            sae_output = sae(activation)
        sae_activation = sae_output["feature_acts"]
        topk_values, topk_indices = torch.topk(sae_activation, topk, dim=-1)
        topk_values = topk_values[0,0] # Get out of batch dimension and seq dimension
        topk_indices = topk_indices[0,0]
        for latent_id, latent_val in zip(topk_indices, topk_values):
            table.at[latent_id.item(), token_id] = latent_val.item()
    return table

def aggregate_activations(acts, aggregate_function):
    if aggregate_function == "mean":
        return acts.mean(dim=-2)
    elif aggregate_function == "max":
        return acts.max(dim=-2).values
    else:
        raise ValueError(f"Unknown aggregation function: {aggregate_function}")

def train_classifier(pretrained_sae, classifier, activation_store, cfg):
    optimizer = torch.optim.Adam(classifier.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
    criterion = torch.nn.CrossEntropyLoss()
    
    wandb_run = init_wandb(cfg)
    testset_acts, testset_labels = activation_store.get_testset_activations()
    i = 0
    while activation_store.has_next():
        acts, labels = activation_store.next_batch()
        
        input_to_classifier = aggregate_activations(acts, cfg["aggregate_function"])
        if not cfg["baseline"]:
            with torch.no_grad():
                sae_output = pretrained_sae(acts)
            # aggregate along the sequence dimension
            input_to_classifier = aggregate_activations(sae_output["feature_acts"], cfg["aggregate_function"])
        
        pred = classifier(input_to_classifier)

        loss = criterion(pred, labels)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            input_to_classifier = aggregate_activations(testset_acts, cfg["aggregate_function"])
            
            if not cfg["baseline"]:
                sae_output = pretrained_sae(testset_acts)
                input_to_classifier = aggregate_activations(sae_output["feature_acts"], cfg["aggregate_function"])
            
            testset_logits = classifier(input_to_classifier) 
            testset_predictions = torch.argmax(testset_logits, dim=-1)
            testset_accuracy = (testset_predictions == testset_labels).float().mean()
        
        
        wandb_run.log({"ce_loss": loss.item(), "accuracy": testset_accuracy}, i)
        i+=1

def train_sae_supervised_data(sae, activation_store, model, cfg):
    optimizer = torch.optim.Adam(sae.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
    wandb_run = init_wandb(cfg)
    
    i = 0
    while activation_store.has_next():
        activations, _ = activation_store.next_batch()
        sae_output = sae(activations)
        log_wandb(sae_output, i, wandb_run)
        if i % cfg["perf_log_freq"]  == 0:
            batch_tokens, _ = activation_store.get_batch_tokens_and_labels()
            log_model_performance(wandb_run, i, model, activation_store, sae, batch_tokens=batch_tokens)


        loss = sae_output["loss"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg["max_grad_norm"])
        sae.make_decoder_weights_and_grad_unit_norm()
        optimizer.step()
        optimizer.zero_grad()
        i+=1

    # Save final model state
    save_checkpoint(wandb_run, sae, cfg, i)


def train_sae_unsupervised_data(sae, activation_store, model, cfg):
    num_batches = cfg["num_tokens"] // cfg["batch_size"]
    optimizer = torch.optim.Adam(sae.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
    pbar = range(num_batches)

    wandb_run = init_wandb(cfg)
    
    for i in pbar:
        batch = activation_store.next_batch()
        sae_output = sae(batch)
        log_wandb(sae_output, i, wandb_run)
        if i % cfg["perf_log_freq"]  == 0:
            log_model_performance(wandb_run, i, model, activation_store, sae)

        # Clogs the diretory memory
        # if i % cfg["checkpoint_freq"] == 0:
        #     save_checkpoint(wandb_run, sae, cfg, i)

        loss = sae_output["loss"]
        # Clogs the diretory memory
        #pbar.set_postfix({"Loss": f"{loss.item():.4f}", "L0": f"{sae_output['l0_norm']:.4f}", "L2": f"{sae_output['l2_loss']:.4f}", "L1": f"{sae_output['l1_loss']:.4f}", "L1_norm": f"{sae_output['l1_norm']:.4f}"})
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg["max_grad_norm"])
        sae.make_decoder_weights_and_grad_unit_norm()
        optimizer.step()
        optimizer.zero_grad()

    # Save final model state
    save_checkpoint(wandb_run, sae, cfg, i)
    

def train_sae_group(saes, activation_store, model, cfgs):
    num_batches = cfgs[0]["num_tokens"] // cfgs[0]["batch_size"]
    optimizers = [torch.optim.Adam(sae.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"])) for sae, cfg in zip(saes, cfgs)]
    pbar = tqdm.trange(num_batches)

    wandb_run = init_wandb(cfgs[0])

    batch_tokens = activation_store.get_batch_tokens()

    for i in pbar:
        batch = activation_store.next_batch()
        counter = 0
        for sae, cfg, optimizer in zip(saes, cfgs, optimizers):
            sae_output = sae(batch)
            loss = sae_output["loss"]
            log_wandb(sae_output, i, wandb_run, index=counter)
            if i % cfg["perf_log_freq"]  == 0:
                log_model_performance(wandb_run, i, model, activation_store, sae, index=counter, batch_tokens=batch_tokens)

            if i % cfg["checkpoint_freq"] == 0:
                save_checkpoint(wandb_run, sae, cfg, i)

            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "L0": f"{sae_output['l0_norm']:.4f}", "L2": f"{sae_output['l2_loss']:.4f}", "L1": f"{sae_output['l1_loss']:.4f}", "L1_norm": f"{sae_output['l1_norm']:.4f}"})
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg["max_grad_norm"])
            sae.make_decoder_weights_and_grad_unit_norm()
            optimizer.step()
            optimizer.zero_grad()
            counter += 1
   
    for sae, cfg, optimizer in zip(saes, cfgs, optimizers):
        save_checkpoint(wandb_run, sae, cfg, i)
