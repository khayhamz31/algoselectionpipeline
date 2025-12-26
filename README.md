# Algorithm Selection Meta-Learning Pipeline

This repository contains a reproducible experimental pipeline for evaluating
meta-learning approaches to algorithm selection using traditional meta-features,
Dataset2Vec embeddings, and hybrid combinations on OpenML benchmark suites.

The pipeline evaluates algorithm selection performance by first gathering
performance metrics of base learning algorithms across multiple datasets, and
then training and evaluating meta-models using different types of meta-features.
A single experimental iteration compares meta-models trained on traditional
meta-features, Dataset2Vec meta-features, and their hybrid combination within a
given benchmark suite.

The core experiment can be executed through the notebook
`metalearning_pipeline.ipynb`, which relies on the following scripts:
`datasets.py`, `qualities.py`, `d2v_qualities.py`, `runs.py`,
`metaclassifier.py`, `metaregressor.py`, `regressor_comparison.py`, and
`run_consolidation.py`.

To reproduce plots and results comparing different benchmark suites, execute the
relevant notebooks in the `helper` directory, ensuring that the `suite_id` and
`random_seed` parameters are set appropriately.

## Running the experiment 


# Dataset2Vec implementation
Dataset2Vec is a model developed by Hadi Jomaa that extracts vector representations from datasets. The original repository can be found at https://github.com/hadijomaa/dataset2vec. 

For its use in the algorithm selection pipeline, minor code modifications were made to ensure compatibility with the downstream experiments. No changes were made to the conceptual design of the model. This version of the repository can be found at https://github.com/khayhamz31/d2v_copy. 

Before conducting the experiments, run the first cell in 'metalearning_pipeline.ipynb' to clone the modified repository into  'Dataset2Vec folder' in the project directory. Dataset2Vec implentation is fully compatible with the pipeline.  

