
from utils import get_dataset_class_names
from sae_insights import dataset_classification, run_experiment_loop, run_non_private_baseline, run_experiment_loop_random_SI, dataset_classification_random, create_concept_to_features_dict
from dense_representation import  create_representations_for_classes
datasets = ["yahoo_questions", "yahoo_answers", "ag_news", "dbpedia"]

for dataset_name in datasets:
    print(f"Creating dense representations for {dataset_name}")
    create_representations_for_classes(dataset_name)