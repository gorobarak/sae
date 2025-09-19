from utils import get_dataset_class_names, get_default_privacy_config, input_epsilon_for_mean
from dense_representation import run_experiment_loop
datasets = ["yahoo_questions", "yahoo_answers", "ag_news", "dbpedia"]
privacy_config = get_default_privacy_config()
privacy_config["use_shuffled_DP"] = True
privacy_config["enabled"] = True
num_reps = 200
for dataset in datasets:
    run_experiment_loop(dataset, 
                        input_epsilon_for_mean,
                        privacy_config,
                        wandb_project=dataset+"_classification",
                        num_repetitions=num_reps)

