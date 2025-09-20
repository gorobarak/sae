from utils import get_dataset_class_names, get_default_privacy_config
from dense_representation import run_experiment_loop
datasets = ["yahoo_questions", "yahoo_answers", "ag_news", "dbpedia"]
privacy_config = get_default_privacy_config()
privacy_config["use_shuffled_DP"] = False
privacy_config["enabled"] = True
num_reps = 200
for dataset in datasets:
    run_experiment_loop(dataset, 
                        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        privacy_config,
                        wandb_project=dataset+"_classification",
                        num_repetitions=num_reps)

