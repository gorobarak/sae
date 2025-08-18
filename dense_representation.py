from matplotlib.pyplot import step
import torch
import torch.nn.functional as F
import torch.distributions as distributions
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import os
import sys
from utils import WANDB_PROJECT, class_idx_to_class_name_dbpedia, get_file_suffix, DBPEDIA_CLASS_NAMES
from collections import defaultdict
import wandb
NUMBER_OF_EXAMPLES_IN_CLASS_DBPEDIA = 40000 


def create_representations_for_classes(dataset="fancyzhx/dbpedia_14", num_classes=7, normalize_embedding=False):
    """
    Create a dense representation of the classes in the dataset using the specified embedding model.
    
    Args:
        dataset: The dataset to be embedded.
        embedding_model: The model used for generating embeddings.
        num_classes: The number of classes in the dataset.

    Returns:
        None: The function saves the dense representations to disk.
    """
    batch_size = 256
    DTYPE = torch.float32
    DEVICE = "cuda"

    # Load the dataset and model
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    model.to(DEVICE); model.to(DTYPE)
    dataset = load_dataset(dataset, split='train')

    for i in range(num_classes):
        dense_representation_acc = torch.zeros((model.get_sentence_embedding_dimension()), dtype=DTYPE, device=DEVICE)
        offset = i * NUMBER_OF_EXAMPLES_IN_CLASS_DBPEDIA
        for j in range(offset, offset + NUMBER_OF_EXAMPLES_IN_CLASS_DBPEDIA, batch_size):
           
            # get the current batch
            interval_end = min(j + batch_size, offset + NUMBER_OF_EXAMPLES_IN_CLASS_DBPEDIA)
            batch = dataset[j: interval_end]
            batch_texts = batch['content'] # [batch]
            
            # get model embeddings
            embeddings = model.encode(batch_texts) # [batch, d_model]
            embeddings = torch.tensor(embeddings, device=DEVICE) 
            if normalize_embedding:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # sum across batch dimension
            embeddings = embeddings.sum(dim=0) # [d_model]
            dense_representation_acc += embeddings

        # Divide by number of samples in class to get mean
        dense_representation_acc /= NUMBER_OF_EXAMPLES_IN_CLASS_DBPEDIA
    
        dir_path = f"checkpoints/dense_representations/dbpedia_class_{i}"
        os.makedirs(dir_path, exist_ok=True)
        file_suffix = get_file_suffix(normalize_embedding=normalize_embedding)
        file_name = "dense_representation" + file_suffix + ".pt"
        torch.save(dense_representation_acc.cpu(), os.path.join(dir_path, file_name))
        print(f"Saved dense representation for class {class_idx_to_class_name_dbpedia(i)}", file=sys.stderr)



def rank_concepts(concepts,
                    model,
                    private=True,
                    epsilon=0.1,
                    num_classes=7):
    device = model.device
    lines = []
    top_1_correct = 0
    top_3_correct = 0
    for i in range(num_classes):
        class_name = class_idx_to_class_name_dbpedia(i)
        dir_path = f"checkpoints/dense_representations/dbpedia_class_{i}"
        file_name = "dense_representation_normalize_embedding.pt"
        dense_representation = torch.load(os.path.join(dir_path, file_name))
        dense_representation = dense_representation.to(device)

        if private:
           make_private_embedding(dense_representation, 
                                  num_of_users_in_database=NUMBER_OF_EXAMPLES_IN_CLASS_DBPEDIA, 
                                  epsilon=epsilon)
        
        # generate concept similarity scores
        concept_scores = []
        for concept in concepts:
            concept_embedding = model.encode(concept) #[d_model]
            concept_embedding = torch.tensor(concept_embedding, device=device)
            concept_scores.append(F.cosine_similarity(dense_representation, concept_embedding, dim=0).item())

        # sort concepts by similarity scores
        concept_scores = torch.tensor(concept_scores)
        sorted_indices = torch.argsort(concept_scores, descending=True)
        sorted_concepts = [concepts[i] for i in sorted_indices]
        sorted_scores = concept_scores[sorted_indices]

        # update scores
        if class_name.lower() == sorted_concepts[0].lower():
            top_1_correct += 1
        if class_name in [c.lower() for c in sorted_concepts[:3]]:
            top_3_correct += 1


        # write ranking
        lines.append(f"Ranking of concepts for class {class_name}:\n")
        j = 1
        for concept, score in zip(sorted_concepts, sorted_scores):
            lines.append(f"{j}.{concept}: {score.item():.3f}\n")
            j += 1
        lines.append("\n")
    
    # save to file
    suffix = get_file_suffix(private=private)
    if private:
        suffix += f"_epsilon={epsilon}"
    filename = "concept_ranking_v2" + suffix + ".txt"
    dir_path = "checkpoints/dense_representations"
    os.makedirs(dir_path, exist_ok=True)
    with open(os.path.join(dir_path, filename), "w") as f:
        f.writelines(lines)
    print("written concept ranking to", os.path.join(dir_path, filename), file=sys.stderr)

    # compute accuracy
    top_1_acc = top_1_correct / num_classes
    top_3_acc = top_3_correct / num_classes
    print(f"Top-1 accuracy: {top_1_acc:.3f}, Top-3 accuracy: {top_3_acc:.3f}", file=sys.stderr)
    return top_1_acc, top_3_acc




def make_private_embedding(embedding_vec, num_of_users_in_database, epsilon):
    """
    add laplace noise to each coordinate
    """ 
    d = embedding_vec.shape[0]
    scale = d / (num_of_users_in_database * epsilon)
    laplace_dist = distributions.Laplace(loc=0, scale=scale)
    noise = laplace_dist.sample(embedding_vec.shape)
    noise = noise.to(embedding_vec.device)
    embedding_vec += noise


def run_experiment_loop():
    """
    Runs the concept ranking experiment for DBpedia classes multiple times and averages results.
    """
    NUM_REPETITIONS=100
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    model.to("cuda"); model.to(torch.float32)
    top1_acc_results_dict = defaultdict(list)
    top3_acc_results_dict = defaultdict(list)
    config={
        "num_repetitions": NUM_REPETITIONS,
        "model": "sentence-transformers/all-mpnet-base-v2",
        "num_of_classes_tested": 7
    }
    run = wandb.init(project=WANDB_PROJECT, name="dense_representation", config=config)
    run.define_metric("top1_acc", step_metric="epsilon")
    run.define_metric("top3_acc", step_metric="epsilon")
    run.define_metric("top1_acc_std", step_metric="epsilon")
    run.define_metric("top3_acc_std", step_metric="epsilon")
    for _ in range(NUM_REPETITIONS):
        for epsilon in [0.1, 0.5, 1, 2, 5, 10]:
            top1_acc, top3_acc = rank_concepts(DBPEDIA_CLASS_NAMES, model, private=True, epsilon=epsilon, num_classes=7)
            top1_acc_results_dict[str(epsilon)].append(top1_acc)
            top3_acc_results_dict[str(epsilon)].append(top3_acc)

    # compute results
    for epsilon in [0.1, 0.5, 1, 2, 5, 10]:
        top1_acc_results = torch.tensor(top1_acc_results_dict[str(epsilon)])
        top3_acc_results = torch.tensor(top3_acc_results_dict[str(epsilon)])
        top1_acc_mean = top1_acc_results.mean().item()
        top1_acc_std = top1_acc_results.std().item()
        top3_acc_mean = top3_acc_results.mean().item()
        top3_acc_std = top3_acc_results.std().item()
    
        run.log({"top1_acc": top1_acc_mean,
                   "top1_acc_std": top1_acc_std,
                   "top3_acc": top3_acc_mean,
                   "top3_acc_std": top3_acc_std,
                   "epsilon": epsilon
                   })
        
    