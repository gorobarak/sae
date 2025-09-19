import os
from sae_lens import SAE
from sympy import use
import torch
import torch.distributions as distributions
from collections import defaultdict
from neuronpedia import get_concept_feature_indicies
from utils import WANDB_DEFAULT_PROJECT, class_idx_to_class_name, get_default_privacy_config, get_file_suffix, get_dataset_metadata, get_dataset_class_names, init_wandb
import wandb
import sys
from ShuffleDPNoise import ShuffledDPHistogramSanitizer



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
                  privacy_config,
                  histogram,
                  max_count,
                  general_histogram=None,
                  max_count_general=None,
                  use_idf_weights=True):
    """
    Rank concepts based on their feature counts in the histogram

    Args:
        concepts: list of concept names
        concepts_to_features_dict: dict mapping concept names to their feature indices
        histogram: histogram of shape [d_sae] containing the counts of features
        max_count: the maximum possible count of a feature in the histogram
        general_histogram: a histogram of feature counts over a general corpus
        max_count_general: the maximum possible count of a feature in the general histogram
        private: whether to make the histogram private
        use_shuffled_DP: whether to use shuffled DP or centralized DP mechanism for privacy
        epsilon: privacy budget for differential privacy
        k: the top k used to construct the histogram
        num_tokens_in_sequence: number of tokens in a sequence

    Returns:
        A tuple (ranked_concepts, frequencies, output_epsilon) 
        ranked_concepts: list of concepts sorted by their frequency in the histogram
        frequencies: tensor of concept frequencies sorted in descending order
        output_epsilon: the privacy budget used (same as input epsilon for centralized DP)

    """
    histogram = histogram.float()
    general_histogram = general_histogram.float().to(histogram.device)  

    output_epsilon = privacy_config.get("input_epsilon", -1)
    if privacy_config["enabled"]:
        noisy_histogram, output_epsilon = make_private_histogram(histogram, privacy_config)
        histogram = noisy_histogram
        

    # prepare feature IDF weights
    if use_idf_weights:
        idf_weights = torch.log((max_count_general + 1) / (general_histogram + 1))  # [d_sae]

    # get concept frequencies
    concept_frequencies = []
    for concept in concepts:
        
        feature_indices = concept_to_features_dict[concept]
        feature_frequencies = histogram[feature_indices] / max_count

        if use_idf_weights:
            feature_idf_weights = idf_weights[feature_indices]
            feature_frequencies = feature_frequencies * feature_idf_weights

        concept_frequency = feature_frequencies.sum().item()
        concept_frequencies.append(concept_frequency)
    
    # sort concepts by frequency
    concept_frequencies = torch.tensor(concept_frequencies)
    sorted_indices = torch.argsort(concept_frequencies, descending=True)
    sorted_concepts = [concepts[i] for i in sorted_indices]
    sorted_frequencies = concept_frequencies[sorted_indices]

    return sorted_concepts, sorted_frequencies, output_epsilon


def dataset_classification(dataset,
                           class_names_to_features_dict, 
                           privacy_config,
                           k=3, 
                           layer=24,
                           use_idf_weights=True
                  ):
    """
    Classifies a dataset classes using SAE-Insights
    Assumes there is a histogram saved for every class in the dataset 

    Args:
        dataset: The dataset to classify
        k: the number of top k features used to construct the histograms
        layer: the layer the activations are taken from
        private: whether to use private histograms
        use_shuffled_DP: whether to use shuffled DP or centralized DP mechanism for privacy
        epsilon: the privacy budget for private histograms
        use_idf_weights: whether to use IDF weights when ranking concepts
    
    Returns:
        A tuple (top1_acc, top3_acc, output_epsilon)
        top1_acc: the top-1 accuracy of the classification
        top3_acc: the top-3 accuracy of the classification
        output_epsilon: the privacy budget used (same as input epsilon for centralized DP)
    """
    lines = []
    
    num_of_classes, class_to_size, _ = get_dataset_metadata(dataset)
    class_names = get_dataset_class_names(dataset)
    num_tokens_in_sequence = 128
    privacy_config["sensitivity"] = num_tokens_in_sequence * k  # each token contributes to k feature occurences in the histogram

    top1_correct = 0
    top3_correct = 0
    for i in range(num_of_classes):

        num_tokens_in_class = class_to_size[i] * num_tokens_in_sequence

        # load 
        dir_path = os.path.join("checkpoints", f"gemma-2-2b_layer_{layer}", dataset,  f"class_{i}")
        filename = "histogram" + get_file_suffix(k=k) + ".pt"
        histogram = torch.load(os.path.join(dir_path, filename), weights_only=True).to('cuda')  # [d_sae]
        dir_path = os.path.join("checkpoints", f"gemma-2-2b_layer_{layer}", "general_histograms")
        filename = "histogram" + get_file_suffix(k=k, dataset="OpenWebText", num_tokens=131072000) + ".pt"
        general_histogram = torch.load(os.path.join(dir_path, filename), weights_only=True).to('cuda')  # [d_sae]
        max_count = 131072000  # maximum possible count of a feature in the general histogram (num_tokens in general corpus)
        
        # Create ranking
        ranked_concepts, frequencies, output_epsilon = rank_concepts(class_names, 
                                                     class_names_to_features_dict, 
                                                     privacy_config,
                                                     histogram,
                                                     num_tokens_in_class, # max possible count of a feature in the histogram
                                                     general_histogram,
                                                     max_count,
                                                     use_idf_weights=use_idf_weights
                                                     )
        
        # update scores:
        if class_names[i] == ranked_concepts[0]:
            top1_correct += 1
        if class_names[i] in ranked_concepts[:3]:
            top3_correct += 1

        # write ranking
        lines.append(f"Ranking of concepts for class {class_names[i]}:\n")
        j = 1
        for concept, frequency in zip(ranked_concepts, frequencies):
            lines.append(f"{j}.{concept}: {frequency.item():.4f}\n")
            j += 1
        lines.append("\n")
    
    # save to file
    filename = "concept_ranking" + get_file_suffix(k=k, private=privacy_config["enabled"], TFIDF=use_idf_weights)
    if privacy_config["enabled"]:
        filename += f"_epsilon={output_epsilon:.2f}"
    filename += ".txt"
    with open(os.path.join("checkpoints", "gemma-2-2b_layer_24", dataset, filename), "w") as f:
        f.writelines(lines)

    print(f"Saved {os.path.join('checkpoints', f"gemma-2-2b_layer_{layer}", dataset, filename)}", file=sys.stderr)
    print(f"Top1 accuracy: {top1_correct / num_of_classes}, Top3 accuracy: {top3_correct / num_of_classes}", file=sys.stderr)

    return (top1_correct / num_of_classes), (top3_correct / num_of_classes), output_epsilon

def create_histograms_for_dataset(dataset, ks=[3], layer=24):
    """
    Create histograms of feature counts of all classes in dataset.
    
    for every token in the class it logs its top k features in a histogram

    Assumes there is activations.pt file in the class directory
    
    Args:
        dataset: The dataset to create histograms for
        ks: The list of top k values to consider for histograms
        layer: the layer the activations are taken from

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
 
def make_private_histogram(histogram, privacy_config):
    if privacy_config["use_shuffled_DP"]:
        shuffled_dp_sanitizer = ShuffledDPHistogramSanitizer()
        noisy_histogram, output_epsilon = shuffled_dp_sanitizer.sanitize(histogram,
                                                                         n=-1,  # Not used in the current implementation
                                                                         k=privacy_config["sensitivity"],
                                                                         epsilon=privacy_config["input_epsilon"],
                                                                         use_sparsity=privacy_config["use_sensitivity"],)
    else:
        scale = privacy_config["sensitivity"] / privacy_config["input_epsilon"]
        laplace = distributions.Laplace(loc=0, scale=scale)
        noise = laplace.sample(histogram.shape)
        noise = noise.to(histogram.device)
        noisy_histogram = histogram + noise
        output_epsilon = privacy_config["input_epsilon"]

    return noisy_histogram, output_epsilon

def run_experiment_loop(dataset,
                        epsilons,
                        privacy_config,
                        wandb_project,
                        run_name,
                        use_idf_weights=True, 
                        num_repetitions=200):
    config = {
        "k": "3",
        "layer": 24,
        "num_of_repetitions": num_repetitions,
        "privacy_config": privacy_config,
        "use_idf_weights": use_idf_weights
    }
    run = init_wandb(wandb_project, run_name, config)

    class_names = get_dataset_class_names(dataset)
    class_names_to_features_dict = create_concept_to_features_dict(class_names)

    top1_acc_results_dict = defaultdict(list)
    top3_acc_results_dict = defaultdict(list)
    for _ in range(num_repetitions):
        for epsilon in epsilons:
            privacy_config["input_epsilon"] = epsilon
            top1_acc, top3_acc, output_epsilon = dataset_classification(dataset,
                                            class_names_to_features_dict,
                                            privacy_config,
                                            k=3,
                                            use_idf_weights=use_idf_weights
                                            )
            top1_acc_results_dict[str(output_epsilon)].append(top1_acc)
            top3_acc_results_dict[str(output_epsilon)].append(top3_acc)

    # compute results
    for output_epsilon_str in top1_acc_results_dict.keys():
        top1_acc_results = torch.tensor(top1_acc_results_dict[output_epsilon_str])
        top3_acc_results = torch.tensor(top3_acc_results_dict[output_epsilon_str])
        top1_acc_mean = top1_acc_results.mean().item()
        top1_acc_std = top1_acc_results.std().item()
        top3_acc_mean = top3_acc_results.mean().item()
        top3_acc_std = top3_acc_results.std().item()

        run.log({
            "top1_acc": top1_acc_mean,
            "top3_acc": top3_acc_mean,
            "top1_acc_std": top1_acc_std,
            "top3_acc_std": top3_acc_std,
            "epsilon": float(output_epsilon_str)
        })

    print("Finished running regular experiments", file=sys.stderr)


def run_non_private_baseline(dataset,
                             use_idf_weights=True, 
                             wandb_project=WANDB_DEFAULT_PROJECT, 
                             run_name="sae_insights_non_private"):
    """
    Runs the concept ranking experiment for dataset classes without privacy.
    """
    config = {
        "k": 3,
        "layer": 24,
        "use_idf_weights": use_idf_weights,
    }
    run = wandb.init(project=wandb_project, name=run_name, config=config, reinit="finish_previous")

    class_names = get_dataset_class_names(dataset)
    class_name_to_features_dict = create_concept_to_features_dict(class_names)
    privacy_config = get_default_privacy_config()
    privacy_config["enabled"] = False  # non-private
    top1_acc, top3_acc, _ = dataset_classification(dataset,
                                                class_name_to_features_dict,
                                                privacy_config,
                                                k=3,
                                                layer=24,
                                                use_idf_weights=use_idf_weights
                                                )
    
    print(f"Non-private Top-1 accuracy: {top1_acc:.3f}, Top-3 accuracy: {top3_acc:.3f}", file=sys.stderr)
    run.log({"top1_acc": top1_acc, "top3_acc": top3_acc})


if __name__ == "__main__":
    pass