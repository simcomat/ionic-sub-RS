# 'Ionic Substitution Graph Recommender System: A Pathway to Novel Materials' Repository

This repository hosts the dataset and supporting resources related to the research paper "Ionic Substitution Graph Recommender System: Discovering New Materials" by Elton Ogoshi and co-authors.

## Repository Content

1. Data Files:
   - `data/occupancy_data.json.gz`: This file provides the underlying data from (OQMD v1.5)[https://static.oqmd.org/static/downloads/qmdb__v1_5__102021.sql.gz] used to build the graphs, denoted as $G^T$.
   - `G100.gexf`: This file contains the graph denoted as $G^{100}$. It is the optimal graph used to build the recommender system.
   - `G000.gexf`: This file contains the graph denoted as $G^{0}$. It is the graph used to make the UMAP 2D visualization of the embedding of ions and sites of stable compounds.

2. Model Files:
   - `models/word2vec_100.model`: This Word2Vec model contains the ion and site embeddings. It was trained using the random walks on the $G^{100}$ graph.
   - `models/word2vec_000.model`: The same as above but for T = 0K.
   - `models/decision_tree_100.joblib`: This decision tree model delineates the minimum distance in the embedding of $G^{100}$ to assign an ion-site occupation.
   - `models/decision_tree_000.joblib`: The same as above but for T = 0K.

3. Code Files:
   - `recommender/core.py`: This Python script provides classes for data manipulation and for interacting with the trained models. It is primarily used for recommending ion-site occupations, a key step in generating potentially new stable compounds.

## Demonstrations

- `Getting_recommendations.ipynb`: This Jupyter notebook demonstrates how to utilize the classes defined in `recommender/core.py` to generate ion-site recommendations.

By providing this comprehensive set of data, models, and tools, this repository facilitates the replication and extension of our work. This can lead to further advancements in the field and may spur the discovery of previously unknown stable compounds.

## Description of the Occupancy Data JSON Structure

The provided JSON file represents a complex data structure containing detailed information about various compounds and their Anonymous Motifs (AMs). This structure is a mixture of integers, strings, dictionaries, and lists, capturing various aspects of the compound's properties, attributes and their distribution.

Here is a detailed breakdown of the structure:

### Root Level

- `total_entries`: (int) Total number of entries in the data set.
- `total_AM`: (int) Total number of unique Anonymous Motifs (AMs).
- `total_AM_sites`: (int) Total number of unique AM sites.
- `total_ions`: (int) Total number of unique ions.
- `ions_distribution`: (dict[str, int]) Count distribution of ions in the entire dataset. Each key is the name of an ion and the corresponding value is its count.
- `site_symmetry_distribution`: (dict[str, int]) Count distribution of site symmetries across all AMs. Each key represents a symmetry label and the corresponding value is its count.
- `cnn_dimensions`: (list[str]) List of CrystalNN dimensions' names. 

### Anonymous Motifs (AM)

- `AM_list`: (list[dict]) List of dictionaries, each representing a unique Anonymous Motif (AM). The list is sorted by the number of compounds of each AM. The key attributes of an AM dictionary are as follows:
  - `AM_label`: (str) Label assigned to the AM.
  - `AM_example_structure`: (dict) An example structure of the AM represented as a dictionary. It includes:
    - `structure`: (str) The structure of the AM.
    - `eq_sites`: (list[int]) List of equivalent sites in the structure.
  - `AM_sg`: (int) Space group number of the AM.
  - `nsites`: (int) Total number of sites in the AM.
  - `n_non_equivalent_sites`: (int) Total number of non-equivalent sites in the AM.
  - `AM_has_ambiguous_wyckoff`: (bool) Boolean indicating whether the AM has ambiguous Wyckoff positions.
  - `n_ambiguous_wyckoff`: (int) Number of ambiguous Wyckoff positions in the AM.
  - `ambiguous_wyckoff`: (dict | None) If the AM has ambiguous Wyckoff positions, this is a dictionary representing them. Otherwise, it is `None`.
  - `site_symmetry_distribution`: (dict[str, int]) Distribution of site symmetries within the AM. Each key is a symmetry label and the corresponding value is its count.
  - `n_entries`: (int) Number of entries for the AM in the database.
  - `entries_ids`: (list[int]) List of entry IDs corresponding to the AM.
  - `CNNf`: (list[float]) CrystalNN fingerprint for the AM.
  - `composition_type_distribution`: (dict[str, int]) Distribution of composition types within the AM (unary, binary, ternary, quaternary).
  - `stoichiometry_distribution`: (dict[str, int]) Distribution of generic stoichiometries types within the AM (e.g., A, AB, AB2, ABC, ABC3, ABCD2, etc).


### AM Sites

Within each AM dictionary, there is a `sites` key which is a list of dictionaries, each representing a unique AM site.

- `sites`: (list[dict]) Each dictionary within the list represents a unique AM site with the following attributes:
  - `label`: (str) Label assigned to the site.
  - `multiplicity`: (int) Multiplicity of the site.
  - `symmetry`: (str) Point symmetry of the site.
  - `site_index`: (int) Index of the site.
  - `ambiguous`: (bool) Boolean indicating whether the site is ambiguous.
  - `CNNf_median`: (list[float]) Median CrystalNN fingerprint for the site over all compounds in AM.
  - `CNNf_std`: (list[float]) Standard CrystalNN fingerprint for the site over all compounds in AM.
  - `total_CNNf_std`: (float) Total standard deviation over all dimensions CrystalNN fingerprint for the site.
  - `total_ions`: (int) Total number of ions present at the site.
  - `ion_distribution`: (dict[str, int]) Distribution of ions at the site. Each key is an ion and the corresponding value is its count.
  - `ions_weights`: (dict[str, dict]) Weights for each ion at the site at different conditions. Each key is an ion and the corresponding value is a dictionary representing the ion's weight under various conditions.
      - `b0`: (float) Total weight attributed to the edge connecting the site and the ion at T = 0K.
      - `b100`: (float) Total weight attributed to the edge connecting the site and the ion at T = 100K.
      - `b300`: (float) Total weight attributed to the edge connecting the site and the ion at T = 300K.
  - `n_occupations`: (int) Number of occupations at the site.
  - `occupancy`: (list[dict]) List of dictionaries, each representing a unique ion occupancy at the site over all compounds in the dataset. Each occupancy dictionary includes the following attributes:
    - `ion`: (str) Name of the ion.
    - `id`: (int) OQMD ID of the compound.
    - `site_index`: (int) Index of the site.
    - `e_hull`: (float) Energy above hull of the corresponding compound.
    - `b0`: (float) Weight to be summed to the total weight the edge connecting the site and the ion at T = 0K.
    - `b100`: (float) Weight to be summed to the total weight the edge connecting the site and the ion at T = 100K.
    - `b300`: (float) Weight to be summed to the total weight the edge connecting the site and the ion at T = 300K.
    - `band_gap`: (float) Band gap of the compound.
    - `magmom_pa`: (float) Magnetic moment per atom of the compound.

For a clearer representation, see the JSON structure illustrated as a tree diagram below:

```
Root Level
│
├─ total_entries: int
├─ total_AM: int
├─ total_AM_sites: int
├─ total_ions: int
├─ ions_distribution: dict[str, int]
├─ site_symmetry_distribution: dict[str, int]
├─ cnn_dimensions: list[str]
└─ AM_list: list[dict]
   │
   └─[0]
     ├─ AM_label: str
     ├─ AM_example_structure
     │  ├─ structure: str
     │  └─ eq_sites: list[int]
     ├─ AM_sg: int
     ├─ nsites: int
     ├─ n_non_equivalent_sites: int
     ├─ AM_has_ambiguous_wyckoff: bool
     ├─ n_ambiguous_wyckoff: int
     ├─ ambiguous_wyckoff: dict | None
     ├─ site_symmetry_distribution: dict[str, int]
     ├─ n_entries: int
     ├─ entries_ids: list[int]
     ├─ CNNf: list[float]
     ├─ composition_type_distribution: dict[str, int]
     └─ sites: list[dict]
        │
        └─[0]
          ├─ label: str
          ├─ multiplicity: int
          ├─ symmetry: str
          ├─ site_index: int
          ├─ ambiguous: bool
          ├─ CNNf_median: list[float]
          ├─ CNNf_std: list[float]
          ├─ total_CNNf_std: float
          ├─ total_ions: int
          ├─ ion_distribution: dict[str, int]
          ├─ ions_weights: dict[str, dict]
          │  └─ H: dict[str, float]
          │     ├─ b0: float
          │     ├─ b100: float
          │     └─ b300: float
          ├─ n_occupations: int
          └─ occupancy: list[dict]
             │
             └─[0]
               ├─ ion: str
               ├─ id: int
               ├─ site_index: int
               ├─ e_hull: float
               ├─ b0: float
               ├─ b100: float
               ├─ b300: float
               ├─ band_gap: float
               └─ magmom_pa: float
```

This tree diagram illustrates how you can access any level of the data. For example, to access the T=0K weight attributed for the ion H at the first site of the first AM , you would navigate as follows: `root['AM_list'][0]['sites'][0]['ions_weights']['H']['b0']`.
