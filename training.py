import torch
import tqdm
from logs import init_wandb, log_wandb, log_model_performance, save_checkpoint



def train_classifier(pretrained_sae, classifier, activation_store, cfg):
    optimizer = torch.optim.Adam(classifier.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
    # optimizer = torch.optim.SGD(classifier.parameters(), lr=cfg["lr"], momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    
    wandb_run = init_wandb(cfg)
    # testset_acts, testset_labels = activation_store.get_testset_activations()
    i = 0
    while activation_store.has_next():
        acts, aggregate_activations, labels = activation_store.next_batch()
        
        input_to_classifier = aggregate_activations
        if not cfg["baseline"]:
            with torch.no_grad():
                sae_output = pretrained_sae(acts)
            # aggregate along the sequence dimension
            if cfg["aggregate_function"] == "mean":
                input_to_classifier = sae_output["feature_acts"].mean(dim=-2)
            elif cfg["aggregate_function"] == "max":
                input_to_classifier = sae_output["feature_acts"].max(dim=-2).values
        
        pred = classifier(input_to_classifier)

        loss = criterion(pred, labels)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # with torch.no_grad():
        #     if []
        #     testset_logits = classifier(testset_acts)
        #     testset_predictions = torch.argmax(testset_logits, dim=1)
        #     testset_accuracy = (testset_predictions == testset_labels).float().mean()
        wandb_run.log({"ce_loss": loss.item(), "accuracy": 0}, i)
        i+=1

def train_sae_supervised_data(sae, activation_store, model, cfg):
    optimizer = torch.optim.Adam(sae.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
    wandb_run = init_wandb(cfg)
    
    i = 0
    while activation_store.has_next():
        activations, _, _ = activation_store.next_batch()
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
