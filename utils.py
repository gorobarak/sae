import sys
import wandb
# Global variables
DBPEDIA_CLASS_NAMES=[
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
]
AG_NEWS_CLASS_NAMES = [
    "World",
    "Sports",
    "Business",
    "Science/Technology"
]


NEWSGROUPS_CLASS_NAMES = [
    'Alternative, Atheism',
    'Computer, Graphics',
    'Computer, Operating Systems, Microsoft Windows, Miscellaneous',
    'Computer, System, IBM, PC, Hardware',
    'Computer, System, Mac, Hardware',
    'Computer, Windows, X Window System',
    'Miscellaneous, For-sale',
    'Recreational, Automobiles',
    'Recreational, Motorcycles',
    'Recreational, Sport, Baseball',
    'Recreational, Sport, Hockey',
    'Science, Cryptography',
    'Science, Electronics',
    'Science, Medicine',
    'Science, Space',
    'Social, Religion, Christian',
    'Talk, Politics, Guns',
    'Talk, Politics, Middle East',
    'Talk, Politics, Miscellaneous',
    'Talk, Religion, Miscellaneous'
    ]

YELP_CLASS_NAMES = ["Terrible, Awful, Horrible",
                     "Bad, Poor, Negative",
                     "Okay, Mediocre, Average, Fair",
                     "Good, Nice, Solid, Strong, Positive",
                     "Great, Excellent, Superb"
                     ]

YAHOO_CLASS_NAMES = ["Society & Culture",
                     "Science & Mathematics",
                     "Health",
                     "Education & Reference",
                     "Computers & Internet",
                     "Sports",
                     "Business & Finance",
                     "Entertainment & Music",
                     "Family & Relationships",
                     "Politics & Government"]

WANDB_DEFAULT_PROJECT = "default"

NUM_CLASSES_AG_NEWS = 4
NUM_EXAMPLES_IN_CLASS_AG_NEWS = [30000] * 4
AG_NEWS_HF_PATH = "fancyzhx/ag_news"

NUM_CLASSES_DBPEDIA = 14
NUM_EXAMPLES_IN_CLASS_DBPEDIA = [40000] * 14
DBPEDIA_HF_PATH = "fancyzhx/dbpedia_14"

NUM_CLASSES_NEWSGROUPS = 20
NEWSGROUPS_NUM_EXAMPLES_IN_CLASS = [480, 584, 591, 590, 578, 
                                    593, 585, 594, 598, 597, 
                                    600, 595, 591, 594, 593, 
                                    599, 546, 564, 465, 377]
NEWSGROUPS_HF_PATH = "SetFit/20_newsgroups"

NUM_CLASSES_YELP = 5
NUM_EXAMPLES_IN_CLASS_YELP = [130000] * 5
YELP_HF_PATH = "Yelp/yelp_review_full"

YAHOO_NUM_CLASSES = 10
YAHOO_NUM_EXAMPLES_IN_CLASS = [140000] * 10
YAHOO_HF_PATH = "community-datasets/yahoo_answers_topics"

def print_and_write(message, lines):
    print(message, file=sys.stderr)
    lines.append(message + "\n")

def class_idx_to_class_name(class_idx, dataset):
    if dataset == "dbpedia":
        return DBPEDIA_CLASS_NAMES[class_idx]
    elif dataset == "ag_news":
        return AG_NEWS_CLASS_NAMES[class_idx]
    elif dataset == "newsgroups":
        return NEWSGROUPS_CLASS_NAMES[class_idx]
    elif dataset == "yelp":
        return YELP_CLASS_NAMES[class_idx]
    elif dataset == "yahoo_answers" or dataset == "yahoo_questions":
        return YAHOO_CLASS_NAMES[class_idx]
    else:
        raise ValueError("Unknown dataset")

def get_file_prefix(**kwargs):
    prefix = ""
    for key, value in kwargs.items():
        if type(value) is bool:
            if value:
                prefix += f"{key}_"
        if type(value) is int or type(value) is str:
            prefix += f"{key}={value}_"
    return prefix

def get_file_suffix(**kwargs):
    suffix = ""
    for key, value in kwargs.items():
        if type(value) is bool:
            if value:
                suffix += f"_{key}"
        if type(value) is int or type(value) is str:
            suffix += f"_{key}={value}"
    return suffix

def get_text_column_name(column_names):
    if "text" in column_names:
        return "text"
    elif "content" in column_names:
        return "content"
    else:
        raise ValueError("No text column found")


def get_dataset_metadata(dataset_name):
    if dataset_name == "dbpedia":
        return NUM_CLASSES_DBPEDIA, NUM_EXAMPLES_IN_CLASS_DBPEDIA, DBPEDIA_HF_PATH
    elif dataset_name == "ag_news":
        return NUM_CLASSES_AG_NEWS, NUM_EXAMPLES_IN_CLASS_AG_NEWS, AG_NEWS_HF_PATH
    elif dataset_name == "newsgroups":
        return NUM_CLASSES_NEWSGROUPS, NEWSGROUPS_NUM_EXAMPLES_IN_CLASS, NEWSGROUPS_HF_PATH
    elif dataset_name == "yelp":
        return NUM_CLASSES_YELP, NUM_EXAMPLES_IN_CLASS_YELP, YELP_HF_PATH
    elif dataset_name == "yahoo_answers" or dataset_name == "yahoo_questions":
        return YAHOO_NUM_CLASSES, YAHOO_NUM_EXAMPLES_IN_CLASS, YAHOO_HF_PATH
    else:
        raise ValueError("Unknown dataset")
    
def get_dataset_class_names(dataset):
    if dataset == "dbpedia":
        return DBPEDIA_CLASS_NAMES
    elif dataset == "ag_news":
        return AG_NEWS_CLASS_NAMES
    elif dataset == "newsgroups":
        return NEWSGROUPS_CLASS_NAMES
    elif dataset == "yelp":
        return YELP_CLASS_NAMES
    elif dataset == "yahoo_answers" or dataset == "yahoo_questions":
        return YAHOO_CLASS_NAMES
    else:
        raise ValueError("Unknown dataset")

def get_label_column(dataset):
    if dataset == "dbpedia":
        return "label"
    elif dataset == "ag_news":
        return "label"
    elif dataset == "newsgroups":
        return "label"
    elif dataset == "yelp":
        return "label"
    elif dataset == "yahoo_answers" or dataset == "yahoo_questions":
        return "topic"
    else:
        raise ValueError("Unknown dataset")
    
def prepare_batch_texts(batch, dataset):
    if dataset == "yahoo_questions":
        batch_texts = [f"{title}\n{content}" for title, content in zip(batch["question_title"], batch["question_content"])]
    elif dataset == "yahoo_answers":
        batch_texts = batch['best_answer']
    else:
        text_column = get_text_column_name(batch.keys())
        batch_texts = batch[text_column]
    return batch_texts

def init_wandb(project, name, config):
    run = wandb.init(project=project, name=name, config=config, reinit="finish_previous")
    run.define_metric("top1_acc", step_metric="epsilon")
    run.define_metric("top3_acc", step_metric="epsilon")
    run.define_metric("top1_acc_std", step_metric="epsilon")
    run.define_metric("top3_acc_std", step_metric="epsilon")
    return run