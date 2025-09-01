import os
from random import sample
import re
from sae_lens import SAE
import torch
import torch.distributions as distributions
from collections import defaultdict
from neuronpedia import get_concept_feature_indicies
from utils import WANDB_DEFAULT_PROJECT, class_idx_to_class_name, get_file_suffix, get_dataset_metadata, get_dataset_class_names, init_wandb
import wandb
import sys



def create_concept_to_features_dict(concept_list, model_id="gemma-2-2b", neuronpedia_sae_id="24-gemmascope-res-16k"):
    concept_to_features_dict = {}
    for concept in concept_list:
        concept_to_features_dict[concept] = get_concept_feature_indicies(concept, model_id=model_id, neuronpedia_sae_id=neuronpedia_sae_id)
    return concept_to_features_dict

def create_histograms(activations_cache,
                     sae, 
                     ks=[3],
                     random=False,
                     sample_size=64):
    """
    Create a top-k features histogram for the activations_cache
    If random is true histogram will consider only sample_size of the tokens in a sequence

    Args:
        activations_cache: torch tensor of shape [num_samples, seq_len, d_model]
        sae: the SAE model
        ks: list of integers, each representing the number of top k features to consider per token
        random: whether to randomly select tokens to consider
        sample_size: the subset size of tokens to consider if random is true

    Returns:
        A dict of histograms {k:histogram} of shape [d_sae] containing the counts of the top-k features in the activations_cache
    """
    histograms = {}
    for k in ks:
        histograms[k] = torch.zeros(sae.cfg.d_sae, dtype=torch.long, device='cuda')

    NUM_SAMPLES_IN_CLASS = activations_cache.shape[0]
    minibatch_size = 1024
    for i in range(0, NUM_SAMPLES_IN_CLASS, minibatch_size):
        mini_batch = activations_cache[i:i+minibatch_size, :, :]  # [minibatch, seq_len, d_model]
        mini_batch = mini_batch.cuda()  # Move to GPU before mulplying with SAE encoder matrix
        with torch.no_grad():
            dict_activations = mini_batch @ sae.W_enc  # [minibatch, seq_len, d_sae]

        # Randomly select tokens to consider
        if random:
            # Get random indices for each sample
            idxs = torch.stack([torch.randperm(mini_batch.size(1))[:sample_size] for _ in range(mini_batch.size(0))]) # [minibatch , sample_size]
            idxs = idxs.unsqueeze(-1).expand(-1, -1, dict_activations.size(-1))  # [minibatch, sample_size, d_sae]
            idxs = idxs.cuda() 
            dict_activations = torch.gather(dict_activations, 1, idxs) # [minibatch, sample_size, d_sae]
            # seq_len = sample size

        # Get top-k features for each token
        max_k = max(ks)
        top_max_k_features = torch.topk(dict_activations, k=max_k, dim=-1) # nametuple(values, indicies)
        top_max_k_features_indices = top_max_k_features.indices  # [minibatch, seq_len, max_k] 
        for k in ks:                    
            top_k_features_indices = top_max_k_features_indices[..., :k]  # [minibatch, seq_len, k] 
            top_k_features_indices = top_k_features_indices.reshape(-1)  # Flatten to [minibatch * seq_len * k] 
            # Count frequency of each feature index in the top-k features and add to histogram
            histograms[k] += torch.bincount(top_k_features_indices, minlength=sae.cfg.d_sae)

    return histograms

def rank_concepts(concepts, 
                  concept_to_features_dict,  
                  histogram, 
                  private=False, 
                  epsilon=0.1, 
                  k=3, 
                  num_tokens_in_sequence=128):
    """
    Rank concepts based on their feature counts in the histogram

    Args:
        concepts: list of concept names
        concepts_to_features_dict: dict mapping concept names to their feature indices
        histogram: histogram of shape [d_sae] containing the counts of features
        private: whether to make the histogram private
        epsilon: privacy budget for differential privacy
        k: the top k used to construct the histogram
        num_tokens_in_sequence: number of tokens in a sequence

    Returns:
        A tuple of (ranked_concepts, frequencies) where each is a list of length k

    """
    histogram = histogram.float()
    
    if private:
        make_private_histogram(histogram, epsilon=epsilon, sensitivity=k*num_tokens_in_sequence)
    
    # get concept frequencies
    concept_frequencies = []
    for concept in concepts:
        concept_feature_indices = concept_to_features_dict[concept]
        concept_frequency = histogram[concept_feature_indices].sum().item()
        concept_frequencies.append(concept_frequency)
    
    # sort concepts by frequency
    concept_frequencies = torch.tensor(concept_frequencies)
    sorted_indices = torch.argsort(concept_frequencies, descending=True)
    sorted_concepts = [concepts[i] for i in sorted_indices]
    sorted_frequencies = concept_frequencies[sorted_indices]

    return sorted_concepts, sorted_frequencies

def dataset_classification_random(dataset, 
                                  class_names_to_features_dict, 
                                  sample_size_portion, 
                                  sae, 
                                  k=3, 
                                  private=False, 
                                  epsilon=0.1):
    """
    Classifies a dataset's classes using random histograms.
    Assumes that the dataset activations are cached.

    Args:
        dataset: The dataset to classify.
        class_names_to_features_dict: A dictionary mapping class names to their feature indices.
        sample_size_portion: The portion of the sequence to sample for the creation of the histogram (between 0 and 1).
        sae: The SAE model
        k: top k used to build histogram
        private: Whether to use differential privacy.
        epsilon: Privacy budget for differential privacy.

    Returns:
        top1_acc, top3_acc
    """
    assert 0 < sample_size_portion <= 1, "sample_size_portion must be between 0 and 1"
    num_classes, class_to_size, _ = get_dataset_metadata(dataset)
    class_names = get_dataset_class_names(dataset)
    lines = []
    
    top1_correct = 0
    top3_correct = 0
    for class_idx in range(num_classes):
       
        # create hisogram
        path = f"checkpoints/gemma-2-2b_layer_24/{dataset}/class_{class_idx}/activations.pt"
        activations = torch.load(path)  # [class_size, seq_len, d_model]
        sample_size = int(activations.size(1) * sample_size_portion)
        histograms = create_histograms(activations, 
                                      sae, 
                                      ks=[k], 
                                      random=True, 
                                      sample_size=sample_size)
        histogram = histograms[k]
        print(f"Created random histogram instance for class {class_names[class_idx]}")
        
        # classify it
        ranked_classes, frequencies = rank_concepts(class_names, 
                                                    class_names_to_features_dict, 
                                                    histogram, 
                                                    private=private,
                                                    epsilon=epsilon,
                                                    k=k,
                                                    num_tokens_in_sequence=sample_size)
        
        # update scores
        if class_names[class_idx] == ranked_classes[0]:
            top1_correct += 1
        if class_names[class_idx] in ranked_classes[:3]:
            top3_correct += 1

        # write ranking
        lines.append(f"Ranking of concepts for class {class_names[class_idx]}:\n")
        j = 1
        for concept, freq in zip(ranked_classes, frequencies):
            lines.append(f"{j}.{concept}: {freq.item()} ({(freq.item() / (class_to_size[class_idx] * sample_size * k) ) * 100:.2f}%)\n")
            j += 1
        lines.append("\n")
    
    # save to file:
    path = f"checkpoints/gemma-2-2b_layer_24/{dataset}"
    filename = "concept_ranking_random" + get_file_suffix(k=k, private=private)
    if private:
        filename += f"_epsilon={epsilon}"
    filename += ".txt"
    with open(os.path.join(path, filename), "w") as f:
        f.writelines(lines)
    
    top1_acc = top1_correct / num_classes
    top3_acc = top3_correct / num_classes

    print(f"saved class ranking to {os.path.join(path, filename)}", file=sys.stderr)
    print(f"Top1 accuracy: {top1_acc}, Top3 accuracy: {top3_acc}", file=sys.stderr)
    return top1_acc, top3_acc

def dataset_classification(dataset,
                           class_names_to_features_dict, 
                           k=3, 
                           layer=24,
                           private=True,
                           epsilon=0.5,
                  ):
    """
    Classifies a dataset classes using SAE-Insights
    Assumes there is a histogram saved for every class in the dataset 

    Args:
        concepts: A list of concepts to test against
        dataset: The dataset to classify
        k: the number of top k features used to construct the histograms
        layer: the layer the activations are taken from
        private: whether to use private histograms
        epsilon: the privacy budget for private histograms
    """
    lines = []
    
    num_of_classes, class_to_size, _ = get_dataset_metadata(dataset)
    class_names = get_dataset_class_names(dataset)
    num_tokens_in_sequence = 127
    
    top1_correct = 0
    top3_correct = 0
    for i in range(num_of_classes):

        num_tokens_in_class = class_to_size[i] * num_tokens_in_sequence

        # load histogram
        dir_path = os.path.join("checkpoints", f"gemma-2-2b_layer_{layer}", dataset,  f"class_{i}")
        filename = "histogram" + get_file_suffix(k=k) + ".pt"
        histogram = torch.load(os.path.join(dir_path, filename)).to('cuda')  # [d_sae]
        
        # Create ranking
        ranked_concepts, frequencies = rank_concepts(class_names, 
                                                     class_names_to_features_dict, 
                                                     histogram, 
                                                     private=private, 
                                                     epsilon=epsilon, 
                                                     k=k, 
                                                     num_tokens_in_sequence=num_tokens_in_sequence)
        
        # update scores:
        if class_names[i] == ranked_concepts[0]:
            top1_correct += 1
        if class_names[i] in ranked_concepts[:3]:
            top3_correct += 1

        # write ranking
        lines.append(f"Ranking of concepts for class {class_names[i]}:\n")
        j = 1
        for concept, frequency in zip(ranked_concepts, frequencies):
            lines.append(f"{j}.{concept}: {frequency.item()} ({(frequency.item() / (num_tokens_in_class * k) ) * 100:.2f}%)\n")
            j += 1
        lines.append("\n")
    
    # save to file
    filename = "concept_ranking" + get_file_suffix(k=k, private=private)
    if private:
        filename += f"_epsilon={epsilon}"
    filename += ".txt"
    with open(os.path.join("checkpoints", "gemma-2-2b_layer_24", dataset, filename), "w") as f:
        f.writelines(lines)

    print(f"Saved {os.path.join('checkpoints', f"gemma-2-2b_layer_{layer}", dataset, filename)}", file=sys.stderr)
    print(f"Top1 accuracy: {top1_correct / num_of_classes}, Top3 accuracy: {top3_correct / num_of_classes}", file=sys.stderr)

    return (top1_correct / num_of_classes), (top3_correct / num_of_classes)

def create_histograms_for_dataset(dataset, ks=[3], layer=24):
    """
    Create histograms of feature counts for every class in dataset
    For every token in the class it logs its top k features in a histogram
    Histograms are saved to file
    Assumes there is activations.pt file in the class directory
    
    Args:
    - dataset: The dataset to create histograms for
    - ks: The list of top k values to consider for histograms
    - layer: the layer the activations are taken from

    Returns:
        None; saves histogram to file


    """
    assert layer in range(0,26), "layer must be in range 0-25"
    model_name = "gemma-2-2b"
    sae_release = "gemma-scope-2b-pt-res-canonical"
    sae_id = f"layer_{layer}/width_16k/canonical" 
    sae = SAE.from_pretrained(sae_release, sae_id, device="cuda")[0]
    num_classes, _, _ = get_dataset_metadata(dataset)

    for class_idx in range(num_classes):
        dir_path = os.path.join("checkpoints", f"{model_name}_layer_{layer}", dataset, f"class_{class_idx}")

        # load activations
        activations = torch.load(os.path.join(dir_path, "activations.pt")) # [batch, seq, d_model] on CPU
         
        # Filter bos token
        activations = activations[:, 1:, :]  # [batch, seq_len - 1, d_model]

        # Create histograms
        histograms = create_histograms(activations, sae, ks=ks, random=False)  # {k: histogram}
        
        # Save histograms
        for k, histogram in histograms.items():
            file_suffix = get_file_suffix(k=k)
            filename = "histogram" + file_suffix + ".pt"
            torch.save(histogram, os.path.join(dir_path, filename))
            print(f"Histogram saved to {os.path.join(dir_path, filename)}", file=sys.stderr)
 
def make_private_histogram(histogram, epsilon, sensitivity):
    scale = sensitivity / epsilon
    laplace = distributions.Laplace(loc=0, scale=scale)
    noise = laplace.sample(histogram.shape)
    noise = noise.to(histogram.device)
    histogram += noise

def run_experiment_loop(dataset, 
                        epsilons, 
                        wandb_project=WANDB_DEFAULT_PROJECT, 
                        num_repetitions=100):
    config = {
        "k": "3",
        "layer": 24,
        "num_of_repetitions": num_repetitions
    }
    run = init_wandb(wandb_project, "sae_insights", config)
    
    class_names = get_dataset_class_names(dataset)
    class_names_to_features_dict = create_concept_to_features_dict(class_names)

    top1_acc_results_dict = defaultdict(list)
    top3_acc_results_dict = defaultdict(list)
    for _ in range(num_repetitions):
        for epsilon in epsilons:
            top1_acc, top3_acc = dataset_classification(dataset,
                                            class_names_to_features_dict,
                                            k=3,
                                            private=True,
                                            epsilon=epsilon
                                            )
            top1_acc_results_dict[str(epsilon)].append(top1_acc)
            top3_acc_results_dict[str(epsilon)].append(top3_acc)
    
    # compute results
    for epsilon in epsilons:
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

    print("Finished running regular experiments", file=sys.stderr)

def run_experiment_loop_random_SI(dataset, 
                        epsilons, 
                        wandb_project=WANDB_DEFAULT_PROJECT, 
                        num_repetitions=100, 
                        sample_size_portion=0.5):
    config = {
        "k": "3",
        "layer": 24,
        "num_of_repetitions": num_repetitions,
        "sample_size_portion": sample_size_portion  
    }
    
    run = init_wandb(wandb_project, "sae_insights_random", config)

    # load sae
    sae_release = "gemma-scope-2b-pt-res-canonical"
    sae_id = f"layer_{24}/width_16k/canonical" 
    sae = SAE.from_pretrained(sae_release, sae_id, device="cuda")[0]

    # load concepts to features dict
    class_names = get_dataset_class_names(dataset)
    class_names_to_features_dict = create_concept_to_features_dict(class_names)

    # start experiments
    top1_acc_results_dict = defaultdict(list)
    top3_acc_results_dict = defaultdict(list)
    for _ in range(num_repetitions):
        for epsilon in epsilons:
            top1_acc, top3_acc = dataset_classification_random(dataset,
                                            class_names_to_features_dict,
                                            sample_size_portion=sample_size_portion,
                                            sae=sae,
                                            k=3,
                                            private=True,
                                            epsilon=epsilon
                                            )
            top1_acc_results_dict[str(epsilon)].append(top1_acc)
            top3_acc_results_dict[str(epsilon)].append(top3_acc)

    # compute results
    for epsilon in epsilons:
        top1_results = torch.tensor(top1_acc_results_dict[str(epsilon)])
        top3_results = torch.tensor(top3_acc_results_dict[str(epsilon)])
        top1_mean = top1_results.mean().item()
        top1_std = top1_results.std().item()
        top3_mean = top3_results.mean().item()
        top3_std = top3_results.std().item()

        run.log({
            "top1_acc": top1_mean,
            "top3_acc": top3_mean,
            "top1_acc_std": top1_std,
            "top3_acc_std": top3_std,
            "epsilon": epsilon
        })

    print("finish running random SI experiments", file=sys.stderr)

def run_non_private_baseline(concepts, dataset, wandb_project=WANDB_DEFAULT_PROJECT):
    """
    Runs the concept ranking experiment for dataset classes without privacy.
    """
    config = {
        "k": 3,
        "layer": 24
    }
    run = wandb.init(project=wandb_project, name="sae_insights_non_private", config=config, reinit="finish_previous")
    concept_to_features_dict = create_concept_to_features_dict(concepts)
    top1_acc, top3_acc = dataset_classification(concepts, concept_to_features_dict, dataset, k=3, layer=24, private=False)
    print(f"Non-private Top-1 accuracy: {top1_acc:.3f}, Top-3 accuracy: {top3_acc:.3f}", file=sys.stderr)
    run.log({"top1_acc": top1_acc, "top3_acc": top3_acc})


if __name__ == "__main__":
    pass