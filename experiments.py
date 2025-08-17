import os
os.environ["HF_HOME"] = "/home/yandex/APDL2425a/group_12/gorodissky/.cache/huggingface"
from sae_lens import SAE
import torch
import torch.distributions as distributions
from transformer_lens import HookedTransformer
from neuronpedia import get_description, get_concept_feature_indicies, TOPICS
from utils import class_idx_to_class_name_dbpedia, get_file_prefix, get_file_suffix
import wandb



# Load feature indexes and values for all classes
def load_features(model_id, layer, reactivations=False, matrix_multiply=True):
    file_prefix = get_file_prefix(reactivations=reactivations, matrix_multiply=matrix_multiply)
    feature_indeices = []
    feature_values = []
    for i in range(7):
        
        dir_path = os.path.join("checkpoints", f"{model_id}_layer_{layer}", f"dbpedia_class_{i}")
        
        file_name = file_prefix + "features_indicies.pt"
        with open(os.path.join(dir_path, file_name), "rb") as f:
            feature_indeices.append(torch.load(f))
        file_name = file_prefix + "features_values.pt"
        with open(os.path.join(dir_path, file_name), "rb") as f:
            feature_values.append(torch.load(f))
    
    return feature_indeices, feature_values
    


# Get descriptions for features and write to file
def get_descriptions(feature_indeices, feature_values, model_id, neuronpedia_sae_id):
    num_classes = len(feature_indeices)
    lines = []
    for i in range(num_classes):
        lines.append(f"Top features for class {class_idx_to_class_name_dbpedia(i)}:\n")
        for feature_idx, feature_value in zip(feature_indeices[i], feature_values[i]):
            description = get_description(feature_idx, model_id, neuronpedia_sae_id)
            if description:
                lines.append(f"Feature {feature_idx}:\n")
                lines.append(f"{feature_value.item()} -- {description}\n")
        lines.append("\n")    
    return lines


#### Experiment: What is the most popular concept, tested against a list of concepts?
####             Every class is constructed a frequency histogram of the top features across a subset of the class tokens                
#### Motivation: Check if the concept with the highest frequency is the correct one
def rank_concepts(concept_list, k=20, 
                  num_of_classes=7, 
                  layer=24,
                  private=True,
                  epsilon=0.5,
                  save_filename="concept_ranking"):

    lines = []
    NUM_SAMPLES_IN_CLASS = 40000
    NUM_OF_TOKENS_SEQUENCE = 127
    NUM_OF_TOKENS_IN_CLASS = NUM_SAMPLES_IN_CLASS * NUM_OF_TOKENS_SEQUENCE
    concept_to_feature_indices = {}
    for concept in concept_list:
        concept_to_feature_indices[concept] = get_concept_feature_indicies(concept, model_id="gemma-2-2b", neuronpedia_sae_id="24-gemmascope-res-16k")

    top1_correct = 0
    top3_correct = 0
    for i in range(num_of_classes):
        # load histogram
        dir_path = os.path.join("checkpoints", f"gemma-2-2b_layer_{layer}", f"dbpedia_class_{i}")
        filename = "histogram_v2" + get_file_suffix(k=k) + ".pt"
        with open(os.path.join(dir_path, filename), "rb") as f:
            histogram = torch.load(f)
        histogram = histogram.float()
        
        if private:
            make_private_histogram(histogram, epsilon=epsilon, dim=k*NUM_OF_TOKENS_SEQUENCE)
        
        # get concept frequencies
        concept_frequencies = []
        for concept in concept_list:
            concept_feature_indices = concept_to_feature_indices[concept]
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
    
    # write to file
    filename = save_filename + get_file_suffix(k=k, private=private)
    if private:
        filename += f"_epsilon={epsilon}"
    filename += ".txt"
    with open(os.path.join("checkpoints", "gemma-2-2b_layer_24", filename), "w") as f:
        f.writelines(lines)

    return (top1_correct / num_of_classes), (top3_correct / num_of_classes)


 
def make_private_histogram(histogram, epsilon, dim):
    scale = dim / epsilon
    laplace = distributions.Laplace(loc=0, scale=scale)
    noise = laplace.sample(histogram.shape)
    noise = noise.to(histogram.device)
    histogram += noise


concept_list = [
    "Company",
    "Educational institution",
    "Artist",
    "Athlete",
    "Office holder",
    "Mean of transportation",
    "Building",
    "Natural place",
    "Village",
    "Animal",
    "Plant",
    "Album",
    "Film",
    "Written work", 
    "Naval vessel",
]
config = {"k": "3",
          "dataset": "fancyzhx/dbpedia_14"}
run = wandb.init(project="privacy_experiment_whole_class_summarization", name="features_histogram", config=config)
run.define_metric("top1_acc", step_metric="epsilon")
run.define_metric("top3_acc", step_metric="epsilon")
for epsilon in [0.1, 0.5, 1, 2, 5, 10]:
    top1_acc, top3_acc = rank_concepts(concept_list, k=3, num_of_classes=7, layer=24, private=True, epsilon=epsilon, save_filename="concept_ranking")
    print(f"Top-1 accuracy for epsilon={epsilon}: {top1_acc:.2f}")
    print(f"Top-3 accuracy for epsilon={epsilon}: {top3_acc:.2f}")
    run.log({"top1_acc": top1_acc, "top3_acc": top3_acc, "epsilon": epsilon})

for k in [3, 10]:
    rank_concepts(TOPICS, k=k, num_of_classes=7, layer=24, save_filename="ranking_generated_topics_new")

# End of experiment


# for k in [3, 5, 10]:
#     rank_concepts(concept_list, k=k, num_of_classes=7, layer=24, private=False, epsilon=0.1, save_filename="concept_ranking")
