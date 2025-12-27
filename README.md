# Algorithm Selection Meta-Learning Pipeline

This repository contains a reproducible experimental pipeline for evaluating
meta-learning approaches to algorithm selection using traditional meta-features,
Dataset2Vec embeddings, and hybrid combinations on OpenML benchmark suites.

The pipeline first downloads datasets from OpenML according to the selected benchmark suite. It then extracts dataset representations using two approaches: traditional meta-features obtained via the OpenML API, and deep learning–based meta-features computed using the Dataset2Vec model. Next, it collects performance metrics of the base learning algorithms across all datasets. Using these inputs, the pipeline trains and evaluates meta-models for algorithm selection. Each experimental iteration compares meta-models trained on traditional meta-features, Dataset2Vec meta-features, and their hybrid combination within the same benchmark suite.

The core experiment can be executed through the notebook
`metalearning_pipeline.ipynb`, which relies on the following scripts:
`datasets.py`, `qualities.py`, `d2v_qualities.py`, `runs.py`,
`metaclassifier.py`, `metaregressor.py`, `regressor_comparison.py`, and
`run_consolidation.py`.

To reproduce plots and results comparing different benchmark suites, execute the
relevant notebooks in the `helper` directory, ensuring that the `suite_id` and
`random_seed` parameters are set appropriately.

## Running the experiment for a single benchmark suite

1. Install the required dependencies and clone the modified Dataset2Vec
   repository by running the first cell in `metalearning_pipeline.ipynb`.
2. Execute the `metalearning_pipeline.ipynb` notebook from top to bottom.

## Outputs

1. OpenML datasets are stored in `test_datasets`.
2. Meta-features are stored in:
   - `qualities/d2v/metafeatures.csv` (Dataset2Vec)
   - `qualities/traditional/metafeatures.csv` (traditional)
3. Base learner performance metrics are stored in
   `runs/accuracies/{dataset_id}/{base_learner}`.
4. Meta-classifier results are stored in `results/meta_classifier_results`.
5. Meta-regressor results are stored in `results/meta_regressor_results` and
   `results/regresscomparison`.

## Requirements

This project is implemented in Python (≥3.9) and relies on the following core
scientific and machine learning libraries:

- **NumPy**, **Pandas**, **SciPy** — numerical computation and data handling
- **scikit-learn** — preprocessing, baseline models, and meta-models
- **matplotlib**, **seaborn** — visualisation
- **OpenML** — dataset and run retrieval
- **PyMFE** — traditional meta-feature extraction
- **XGBoost** — tree-based learning algorithms
- **TensorFlow** — deep meta-feature extraction (Dataset2Vec)
- **tqdm** — progress monitoring
- **ipywidgets** — interactive benchmark selection in notebooks

All required packages can be installed via:

```bash
pip install -r requirements.txt


## Dataset2Vec implementation

Dataset2Vec is a model developed by Hadi Jomaa that extracts vector
representations from datasets. The original repository can be found at:
https://github.com/hadijomaa/dataset2vec.

For its use in the algorithm selection pipeline, minor code modifications were
made to ensure compatibility with the downstream experiments. No changes were
made to the conceptual design of the model. The modified version of the
repository can be found at:
https://github.com/khayhamz31/d2v_copy.

Before conducting the experiments, run the first cell in
`metalearning_pipeline.ipynb` to clone the modified repository into a
`Dataset2Vec` folder in the project directory. The Dataset2Vec implementation is
fully compatible with the pipeline.
