# Algorithm Selection Meta-Learning Pipeline

This repository contains a reproducible pipeline for evaluating meta-learning
approaches to algorithm selection using traditional meta-features, Dataset2Vec
embeddings, and hybrid combinations on OpenML benchmark suites.

# Dataset2Vec implementation
Dataset2Vec is a model developed by Hadi Jomaa that extracts vector representations from datasets. The original repository can be found at https://github.com/hadijomaa/dataset2vec. 

For its use in the algorithm selection pipeline, minor code modifications were made to ensure compatibility with the downstream experiments. No changes were made to the conceptual design of the model. This version of the repository can be found at https://github.com/khayhamz31/d2v_copy. 

Before conducting the experiments, run the first cell in 'metalearning_pipeline.ipynb' to clone the modified repository into  'Dataset2Vec folder' in the project directory. Dataset2Vec implentation is fully compatible with the pipeline.  

