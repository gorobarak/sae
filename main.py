
from  generate_frequent_directions import main


datasets = ["dbpedia", "ag_news", "yahoo_questions"]

for dataset in datasets:
    main(dataset, tau=5)


