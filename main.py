from sae_insights import dataset_classification, create_concept_to_features_dict
from utils import get_dataset_class_names, get_default_privacy_config

datasets = ["yahoo_answers", "yahoo_questions", "dbpedia", "ag_news"]
privacy_config = get_default_privacy_config()
privacy_config["enabled"] = False
for dataset in datasets:
    class_names = get_dataset_class_names(dataset)
    class_names_to_features = create_concept_to_features_dict(class_names)
    dataset_classification(
        dataset,
        class_names_to_features,
        privacy_config,
    )
