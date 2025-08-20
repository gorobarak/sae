import os
from sae_lens import SAE
import torch
import torch.distributions as distributions
from collections import defaultdict
from neuronpedia import get_description, get_concept_feature_indicies, TOPICS
from utils import class_idx_to_class_name_dbpedia, get_file_suffix, DBPEDIA_CLASS_NAMES, WANDB_PROJECT
import wandb
import sys


def create_histogram(ks=[3], layer=24):
    """
    Create histograms of feature counts for every class in dbpedia 
    Gets the top k features for token in every sequences in the class
    Maintain a histogram of features counts and saves to file
    """
    assert layer in range(0,26), "layer must be in range 0-25"
    model_name = "gemma-2-2b"
    sae_release = "gemma-scope-2b-pt-res-canonical"
    sae_id = f"layer_{layer}/width_16k/canonical" 
    sae = SAE.from_pretrained(sae_release, sae_id, device="cuda")[0]

    for class_idx in range(7, 14):
        dir_path = os.path.join("checkpoints", f"{model_name}_layer_{layer}", f"dbpedia_class_{class_idx}")

        # load activations
        with open(os.path.join(dir_path, "activations.pt"), "rb") as f:
            activations = torch.load(f) # [batch, seq, d_model] on CPU
         
        # Filter bos token
        activations = activations[:, 1:, :]  # [batch, seq_len - 1, d_model]

        NUM_SAMPLES_IN_CLASS = activations.shape[0]
        minibatch_size = 1024
        histograms = {}
        for k in ks:
            histograms[k] = torch.zeros(sae.cfg.d_sae, dtype=torch.long, device='cuda')
        
        for i in range(0, NUM_SAMPLES_IN_CLASS, minibatch_size):
            mini_batch = activations[i:i+minibatch_size, :, :]  # [minibatch_size, seq_len - 1, d_model]
            mini_batch = mini_batch.to('cuda')  # Move to GPU before mulplying with SAE encoder matrix
            with torch.no_grad():
                dict_activations = mini_batch @ sae.W_enc  # [minibatch_size, seq_len - 1, d_sae]
            

            # Get top-k features for each token
            max_k = max(ks)
            top_max_k_features = torch.topk(dict_activations, k=max_k, dim=-1)
            top_max_k_features_indices = top_max_k_features.indices  # [minibatch, seq_len - 1, max_k] 
            for k in ks:                    
                top_k_features_indices = top_max_k_features_indices[..., :k]  # [minibatch, seq_len - 1, k] 
                top_k_features_indices = top_k_features_indices.reshape(-1)  # Flatten to [minibatch * (seq_len - 1) * k] 
                # Count frequency of each feature index in the top-k features and add to histogram
                histograms[k] += torch.bincount(top_k_features_indices, minlength=sae.cfg.d_sae)
        
        # Save histograms
        for k, histogram in histograms.items():
            file_suffix = get_file_suffix(k=k)
            filename = "histogram_v2" + file_suffix + ".pt"
            with open(os.path.join(dir_path, filename), "wb") as f:
                torch.save(histogram, f)
            print(f"Histogram saved to {os.path.join(dir_path, filename)}", file=sys.stderr)


def create_concept_to_features_dict(concept_list, model_id="gemma-2-2b", neuronpedia_sae_id="24-gemmascope-res-16k"):
    concept_to_features_dict = {}
    for concept in concept_list:
        concept_to_features_dict[concept] = get_concept_feature_indicies(concept, model_id=model_id, neuronpedia_sae_id=neuronpedia_sae_id)
    return concept_to_features_dict

def rank_concepts(concept_list,
                  concept_to_features_dict, 
                  k=3, 
                  layer=24,
                  private=True,
                  epsilon=0.5,
                  dataset="dbpedia",
                  ):
    """
    Rank concepts based on their SAE features frequency in the class
    """
    lines = []
    NUM_SAMPLES_IN_CLASS = 40000
    NUM_OF_TOKENS_SEQUENCE = 127
    NUM_OF_TOKENS_IN_CLASS = NUM_SAMPLES_IN_CLASS * NUM_OF_TOKENS_SEQUENCE
    num_of_classes = 14

    top1_correct = 0
    top3_correct = 0
    for i in range(num_of_classes):
        # load histogram
        dir_path = os.path.join("checkpoints", f"gemma-2-2b_layer_{layer}", f"{dataset}_class_{i}")
        filename = "histogram_v2" + get_file_suffix(k=k) + ".pt"
        with open(os.path.join(dir_path, filename), "rb") as f:
            histogram = torch.load(f)
        histogram = histogram.float()
        
        if private:
            make_private_histogram(histogram, epsilon=epsilon, sensitivity=k*NUM_OF_TOKENS_SEQUENCE)
        
        # get concept frequencies
        concept_frequencies = []
        for concept in concept_list:
            concept_feature_indices = concept_to_features_dict[concept]
            concept_frequency = histogram[concept_feature_indices].sum().item()
            concept_frequencies.append(concept_frequency)
        
        # sort concepts by frequency
        concept_frequencies = torch.tensor(concept_frequencies)
        sorted_indices = torch.argsort(concept_frequencies, descending=True)
        sorted_concepts = [concept_list[i] for i in sorted_indices]
        sorted_frequencies = concept_frequencies[sorted_indices]

        # update scores:
        class_name = class_idx_to_class_name_dbpedia(i).lower()
        if class_name == sorted_concepts[0].lower():
            top1_correct += 1
        if class_name in [c.lower() for c in sorted_concepts[:3]]:
            top3_correct += 1

        # write ranking
        lines.append(f"Ranking of concepts for class {class_name}:\n")
        j = 1
        for concept, frequency in zip(sorted_concepts, sorted_frequencies):
            lines.append(f"{j}.{concept}: {frequency.item()} ({(frequency.item() / (NUM_OF_TOKENS_IN_CLASS * k) ) * 100:.2f}%)\n")
            j += 1
        
        lines.append("\n")
    
    # save to file
    filename = f"concept_ranking_{dataset}" + get_file_suffix(k=k, private=private)
    if private:
        filename += f"_epsilon={epsilon}"
    filename += ".txt"
    with open(os.path.join("checkpoints", "gemma-2-2b_layer_24", filename), "w") as f:
        f.writelines(lines)
    
    print(f"Saved {os.path.join('checkpoints', 'gemma-2-2b_layer_24', filename)}", file=sys.stderr)
    print(f"top1 accuracy: {top1_correct / num_of_classes}, top3 accuracy: {top3_correct / num_of_classes}", file=sys.stderr)

    return (top1_correct / num_of_classes), (top3_correct / num_of_classes)


 
def make_private_histogram(histogram, epsilon, sensitivity):
    scale = sensitivity / epsilon
    laplace = distributions.Laplace(loc=0, scale=scale)
    noise = laplace.sample(histogram.shape)
    noise = noise.to(histogram.device)
    histogram += noise


def run_experiment_loop():
    NUM_OF_REPETITIONS = 100
    config = {"k": "3",
              "layer": 24,
            "num_of_repetitions": NUM_OF_REPETITIONS}
    run = wandb.init(project=WANDB_PROJECT, name="sae_insights", config=config, reinit="finish_previous")
    run.define_metric("top1_acc", step_metric="epsilon")
    run.define_metric("top3_acc", step_metric="epsilon")
    run.define_metric("top1_acc_std", step_metric="epsilon")
    run.define_metric("top3_acc_std", step_metric="epsilon")
    top1_acc_results_dict = defaultdict(list)
    top3_acc_results_dict = defaultdict(list)
    concept_to_feature_dict = create_concept_to_features_dict(DBPEDIA_CLASS_NAMES)
    for _ in range(NUM_OF_REPETITIONS):
        for epsilon in [0.1, 0.5, 1, 2, 5, 10]:
            top1_acc, top3_acc = rank_concepts(DBPEDIA_CLASS_NAMES,
                                            concept_to_feature_dict,
                                            k=3,
                                            private=True,
                                            epsilon=epsilon,
                                            dataset="dbpedia")
            top1_acc_results_dict[str(epsilon)].append(top1_acc)
            top3_acc_results_dict[str(epsilon)].append(top3_acc)
    print("Finished running experiments", file=sys.stderr)
    # compute results
    for epsilon in [0.1, 0.5, 1, 2, 5, 10]:
        top1_acc_results = torch.tensor(top1_acc_results_dict[str(epsilon)])
        top3_acc_results = torch.tensor(top3_acc_results_dict[str(epsilon)])
        top1_acc_mean = top1_acc_results.mean().item()
        top1_acc_std = top1_acc_results.std().item()
        top3_acc_mean = top3_acc_results.mean().item()
        top3_acc_std = top3_acc_results.std().item()

        run.log({
            "top1_acc": top1_acc_mean,
            "top3_acc": top3_acc_mean,
            "top1_acc_std": top1_acc_std,
            "top3_acc_std": top3_acc_std,
            "epsilon": epsilon
        })



def run_non_private_baseline():
    """
    Runs the concept ranking experiment for DBpedia classes without privacy.
    """
    config = {
        "k": 3,
        "layer": 24
    }
    run = wandb.init(project=WANDB_PROJECT, name="sae_insights_non_private", config=config, reinit="finish_previous")
    concept_to_features_dict = create_concept_to_features_dict(DBPEDIA_CLASS_NAMES)
    top1_acc, top3_acc = rank_concepts(DBPEDIA_CLASS_NAMES, concept_to_features_dict, k=3, layer=24, private=False)
    print(f"Non-private Top-1 accuracy: {top1_acc:.3f}, Top-3 accuracy: {top3_acc:.3f}", file=sys.stderr)
    run.log({"top1_acc": top1_acc, "top3_acc": top3_acc})

if __name__ == "__main__":
    pass