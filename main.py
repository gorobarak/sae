

from calendar import c
from random import sample
from utils import DBPEDIA_CLASS_NAMES, AG_NEWS_CLASS_NAMES
import wandb
from sae_lens import SAE


if __name__ == "__main__":

    from dense_representation import create_representations_for_classes
    from create_activations_dataset import main
    from sae_insights import create_histograms_for_dataset
   
    dataset = "yahoo_answers"
    create_representations_for_classes(dataset)
    
    main(dataset)
    create_histograms_for_dataset(dataset, ks=[3])

    dataset = "yahoo_questions"
    create_representations_for_classes(dataset)
    
    main(dataset)
    create_histograms_for_dataset(dataset, ks=[3])
