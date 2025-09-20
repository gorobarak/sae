import torch
import torch.nn.functional as F
import torch.distributions as distributions
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import os
import sys
from utils import  WANDB_DEFAULT_PROJECT, init_wandb, get_file_suffix, prepare_batch_texts, get_dataset_metadata, get_label_column, get_dataset_class_names, get_default_privacy_config
from collections import defaultdict
import wandb
from ShuffleDPNoise import ShuffledDPMeanSanitizer



def create_representations_for_classes(dataset_name):
    """
    Create dense representations of the classes in the dataset using the specified embedding model.

    Args:
        dataset: The dataset to be embedded.
        embedding_model: The model used for generating embeddings.

    Returns:
        None: The function saves the mean representations and the tensor of representations to disk for every class.
    """
    batch_size = 256
    DTYPE = torch.float32
    DEVICE = "cuda"

    # Load model
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    model.to(DEVICE); model.to(DTYPE)

    # Load dataset
    num_classes, class_to_size, hf_path = get_dataset_metadata(dataset_name)
    dataset = load_dataset(hf_path, split="train")
    label_column = get_label_column(dataset_name)
    dataset = dataset.sort(label_column)

    class_size_prefix_sum = 0
    for i in range(0, num_classes):
        dense_representation_acc = torch.zeros((model.get_sentence_embedding_dimension()), dtype=DTYPE, device=DEVICE)
        cur_class_size = class_to_size[i]
        embeddings_acc = []
        for j in range(class_size_prefix_sum, class_size_prefix_sum + cur_class_size, batch_size):

            # get the current batch
            interval_end = min(j + batch_size, class_size_prefix_sum + cur_class_size)
            batch = dataset[j: interval_end]
            batch_texts = prepare_batch_texts(batch, dataset_name) # [batch]

            # get model embeddings
            embeddings = model.encode(batch_texts) # [batch, d_model]
            embeddings = torch.tensor(embeddings, device=DEVICE) 
            embeddings = F.normalize(embeddings, p=2, dim=1)
            embeddings_acc.append(embeddings)

            # sum across batch dimension
            embeddings = embeddings.sum(dim=0) # [d_model]
            # accumulate 
            dense_representation_acc += embeddings
        
        # Divide by number of samples in class to get mean
        dense_representation_acc /= cur_class_size
        # Save
        dir_path = os.path.join("checkpoints", "dense_representation", f"{dataset_name}", f"class_{i}")
        os.makedirs(dir_path, exist_ok=True)
        file_name = "dense_representation.pt"
        torch.save(dense_representation_acc.cpu(), os.path.join(dir_path, file_name))
        print(f"Saved dense representation to {os.path.join(dir_path, file_name)}", file=sys.stderr)

        # Concatenate all embeddings for the class
        embeddings_acc = torch.cat(embeddings_acc, dim=0) # [cur_class_size, d_model]
        assert embeddings_acc.size(0) == cur_class_size
        assert embeddings_acc.size(1) == model.get_sentence_embedding_dimension()
        assert len(embeddings_acc.size()) == 2
        # Save
        file_name = "all_representations.pt"
        torch.save(embeddings_acc.cpu(), os.path.join(dir_path, file_name))
        print(f"Saved all representations to {os.path.join(dir_path, file_name)}", file=sys.stderr)

        # Update prefix sum
        class_size_prefix_sum += cur_class_size

def dataset_classification(dataset,
                    model,
                    privacy_config,
                    ):
    device = model.device
    lines = []
    num_classes, class_to_size, _ = get_dataset_metadata(dataset)
    class_names = get_dataset_class_names(dataset)
    
    top_1_correct = 0
    top_3_correct = 0
    for i in range(num_classes):
        dir_path = os.path.join("checkpoints", "dense_representation", dataset, f"class_{i}")
        file_name = "all_representations.pt"
        representations = torch.load(os.path.join(dir_path, file_name))  # [class_size, d_model]
        representations = representations.to(device)

        output_epsilon = None
        dense_representation = None
        if privacy_config["enabled"]:
           privacy_config["sensitivity"] = representations.size(1) / representations.size(0)  # dim / n
           noisy_representation, output_epsilon = make_private_mean(representations, privacy_config)
           dense_representation = noisy_representation
        else:
            dense_representation = torch.mean(representations, dim=0) # [d_model]

        # Classify dense representation 
        ranked_classes, scores = rank_concepts(class_names, dense_representation, model)

        # update scores
        if class_names[i] == ranked_classes[0]:
            top_1_correct += 1
        if class_names[i] in ranked_classes[:3]:
            top_3_correct += 1

        # write ranking
        lines.append(f"Ranking of concepts for class {class_names[i]}:\n")
        j = 1
        for class_name, score in zip(ranked_classes, scores):
            lines.append(f"{j}.{class_name}: {score.item():.3f}\n")
            j += 1
        lines.append("\n")
    
    # save to file
    suffix = get_file_suffix(private=privacy_config["enabled"])
    if privacy_config["enabled"]:
        suffix += f"_epsilon={output_epsilon}"
    filename = "concept_ranking" + suffix + ".txt"
    dir_path = f"checkpoints/dense_representation/{dataset}"
    os.makedirs(dir_path, exist_ok=True)
    with open(os.path.join(dir_path, filename), "w") as f:
        f.writelines(lines)
    print(f"Saved concept ranking to {os.path.join(dir_path, filename)}", file=sys.stderr)

    # compute accuracy
    top_1_acc = top_1_correct / num_classes
    top_3_acc = top_3_correct / num_classes
    print(f"Top-1 accuracy: {top_1_acc:.3f}, Top-3 accuracy: {top_3_acc:.3f}", file=sys.stderr)
    return top_1_acc, top_3_acc, output_epsilon

def rank_concepts(concepts, dense_representation, model):
    concept_scores = []
    for concept in concepts:
        concept_embedding = model.encode(concept) #[d_model]
        concept_embedding = torch.tensor(concept_embedding, device=model.device)
        concept_scores.append(F.cosine_similarity(dense_representation, concept_embedding, dim=0).item())

    # sort concepts by similarity scores
    concept_scores = torch.tensor(concept_scores)
    sorted_indices = torch.argsort(concept_scores, descending=True)
    sorted_concepts = [concepts[i] for i in sorted_indices]
    sorted_scores = concept_scores[sorted_indices]

    return sorted_concepts, sorted_scores


def make_private_mean(representations, privacy_config): 
    
    noisy_representation, output_epsilon = None, None
    
    if privacy_config["use_shuffled_DP"]:
        shuffled_dp_sanitizer = ShuffledDPMeanSanitizer(representations)
        noisy_representation, output_epsilon = shuffled_dp_sanitizer.sanitize(privacy_config["input_epsilon"])
    
    else: # Central DP with Laplace noise
        mean_representation = torch.mean(representations, dim=0) # [d_model]
        scale = privacy_config["sensitivity"] / privacy_config["input_epsilon"]
        laplace_dist = distributions.Laplace(loc=0, scale=scale)
        noise = laplace_dist.sample(mean_representation.shape).to(mean_representation.device)
        noisy_representation = mean_representation + noise
        output_epsilon = privacy_config["input_epsilon"]

    return noisy_representation, output_epsilon

def run_experiment_loop(dataset, 
                        epsilons,
                        privacy_config,  
                        wandb_project, 
                        num_repetitions=100):
    """
    Runs the concept ranking experiment for dataset classes multiple times and averages results.
    """
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    model.to("cuda"); model.to(torch.float32)
    top1_acc_results_dict = defaultdict(list)
    top3_acc_results_dict = defaultdict(list)
    config={
        "num_repetitions": num_repetitions,
        "model": "sentence-transformers/all-mpnet-base-v2",
        "privacy_config": privacy_config
    }
    run = init_wandb(wandb_project, 
                     "dense_representation" + get_file_suffix(DP=privacy_config["use_shuffled_DP"]),
                     config)
    for _ in range(num_repetitions):
        for epsilon in epsilons:
            privacy_config["input_epsilon"] = epsilon
            top1_acc, top3_acc, output_epsilon = dataset_classification(dataset, 
                                                                        model, 
                                                                        privacy_config)
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
    
        run.log({"top1_acc": top1_acc_mean,
                   "top1_acc_std": top1_acc_std,
                   "top3_acc": top3_acc_mean,
                   "top3_acc_std": top3_acc_std,
                   "epsilon": float(output_epsilon_str)
                   })
    print("Finished running experiments", file=sys.stderr)

def run_non_private_baseline(dataset, 
                             wandb_project=WANDB_DEFAULT_PROJECT):
    """
    Runs the concept ranking experiment without privacy.
    """
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    model.to("cuda"); model.to(torch.float32)
    config = {
        "model": "sentence-transformers/all-mpnet-base-v2",
    }
    dummy_privacy_config = get_default_privacy_config()
    dummy_privacy_config["enabled"] = False
    run = wandb.init(project=wandb_project, name="dense_representation_non_private", config=config, reinit="finish_previous")
    top1_acc, top3_acc, _ = dataset_classification(dataset, model, dummy_privacy_config)
    print(f"Non-private Top-1 accuracy: {top1_acc:.3f}, Top-3 accuracy: {top3_acc:.3f}", file=sys.stderr)
    run.log({"top1_acc": top1_acc, "top3_acc": top3_acc})

def hello():
    print("Dense representation")