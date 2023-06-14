# 'Ionic Substitution Graph Recommender System: A Pathway to Novel Materials' Repository

This repository hosts the dataset and supporting resources related to the research paper "Ionic Substitution Graph Recommender System: Discovering New Materials" by Elton Ogoshi and co-authors.

## Repository Content

1. Data Files:
   - `data/occupancy_data.json.gz`: This file provides the underlying data used to build the graphs, denoted as $G^T$.
   - `G100.gexf`: This file contains the graph denoted as $G^{100}$.

2. Model Files:
   - `models/word2vec.model`: This Word2Vec model contains the ion and site embeddings. It was trained using the random walks on the $G^{100}$ graph.
   - `models/decision_tree.joblib`: This decision tree model delineates the minimum distance in the embedding to assign an ion-site occupation.

3. Code Files:
   - `recommender/core.py`: This Python script provides classes for data manipulation and for interacting with the trained models. It is primarily used for recommending ion-site occupations, a key step in generating potentially new stable compounds.

## Demonstrations

- `Getting_recommendations.ipynb`: This Jupyter notebook demonstrates how to utilize the classes defined in `recommender/core.py` to generate ion-site recommendations.

By providing this comprehensive set of data, models, and tools, this repository facilitates the replication and extension of our work. This can lead to further advancements in the field and may spur the discovery of previously unknown stable compounds.