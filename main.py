
from utils import get_dataset_class_names
from sae_insights import run_experiment_loop, run_non_private_baseline, run_experiment_loop_random_SI, dataset_classification_random, create_concept_to_features_dict
from sae_lens import SAE
datasets = ["dbpedia", "ag_news", "yahoo_answers"]
wandb_projects = ["dbpedia_classification", "ag_news_classification", "yahoo_answers_classification"]
for dataset, wandb_project in zip(datasets, wandb_projects):
    num_reps = 150
    eps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] 
    config = {"class_size": 10000 }
    run_experiment_loop_random_SI(dataset,
                                eps,
                                wandb_project=wandb_project,
                                num_repetitions=num_reps,
                                sample_size_portion=0.5,
                                config=config)

