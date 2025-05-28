import heapq
from numpy import half
import torch
import tqdm
from logs import init_wandb, log_wandb, log_model_performance, save_checkpoint
import pandas as pd
from collections import defaultdict

def create_batch_tokens_and_texts(model, dataset, batch_size=16, duplicate_tokens=False):
    all_tokens = []
    all_texts = []
    half_ctx_size = model.cfg.n_ctx // 2
    default_token = torch.tensor([[model.tokenizer.bos_token_id]], device=model.cfg.device)
    for i in range(batch_size):
        sample = next(dataset)
        text = sample["text"]
        tokens = model.to_tokens(text, truncate=True, move_to_device=True, prepend_bos=False)
        if tokens.shape[-1] > half_ctx_size:
            tokens = tokens[:, :half_ctx_size]
        else:
            while tokens.shape[-1] < half_ctx_size:
                tokens = torch.cat([tokens, default_token], dim=-1)
        text = model.tokenizer.decode(tokens[0])
        if duplicate_tokens:
            tokens = torch.cat([tokens, tokens], dim=-1)
        all_texts.append(text)
        all_tokens.append(tokens)
    batch_tokens = torch.cat(all_tokens, dim=0)
    return batch_tokens, all_texts

def get_top_activating_samples(model, sae, cfg, dataset, duplicate_tokens, k=10):
    """
    Get the top k samples that activate every sae latent
    """

    heaps = defaultdict(list)
    num_batches = cfg["num_sequences"] // cfg["batch_size"]
    for i in range(num_batches):
        
        batch_tokens, batch_texts = create_batch_tokens_and_texts(model, dataset, batch_size=cfg["batch_size"], duplicate_tokens=duplicate_tokens)
        # Get activation
        with torch.no_grad():
            _, cache = model.run_with_cache(
                batch_tokens,
                names_filter=[cfg["hook_point"]],
                stop_at_layer=cfg["hook_point_layer"] + 1,
            )
            activations = cache[cfg["hook_point"]]
            
            if duplicate_tokens:
                # Discard the first copy's activations
                half_ctx_size = model.cfg.n_ctx // 2
                activations = activations[:, half_ctx_size: ,:]

            # Get latent activations
            latents_acts = sae.encode(activations)
            
        # Aggregate latent activation along the sequence dimension
        latents_acts_agg = latents_acts.max(dim=-2).values

        # Update the top k samples for each latent 
        for batch_idx in range(cfg['batch_size']):
            for latent_idx in range(cfg['d_sae']):
                
                latent_activation = latents_acts_agg[batch_idx, latent_idx].item()
                item = (latent_activation, i, batch_texts[batch_idx]) # i is used for tie breaking
            
                if i < k:
                    heapq.heappush(heaps[latent_idx], item)
                else:
                    heapq.heappushpop(heaps[latent_idx], item)
    return heaps


def create_words_to_latents_table(model, sae):
    vocab_size = model.cfg.d_vocab
    dict_size = sae.cfg["dict_size"]
    topk = sae.cfg["top_k"]
    table = pd.DataFrame(0, index=range(dict_size), columns=range(vocab_size), dtype=float)
    prefix = model.to_tokens("The following word is: ", move_to_device=True, prepend_bos=False)
    for token_id in range(vocab_size):
        token_id_tensor = torch.tensor([[token_id]], device=sae.cfg["device"])
        input_tensor = torch.cat([prefix, token_id_tensor], dim=-1)
        with torch.no_grad():
            _, cache = model.run_with_cache(
                input_tensor,
                names_filter=[sae.cfg["hook_point"]],
                stop_at_layer=sae.cfg["layer"] + 1,
            )
            activation = cache[sae.cfg["hook_point"]]
            sae_output = sae(activation)
        sae_activation = sae_output["feature_acts"]
        topk_values, topk_indices = torch.topk(sae_activation, topk, dim=-1)
        topk_values = topk_values[0,-1, :] # Get out of batch dimension and seq dimension
        topk_indices = topk_indices[0,-1, :]
        for latent_id, latent_val in zip(topk_indices, topk_values):
            table.iat[latent_id.item(), token_id] = latent_val.item()
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
