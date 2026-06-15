# ClusterLOCO

[![CI](https://github.com/DataSlingers/ClusterLOCO/actions/workflows/ci.yml/badge.svg)](https://github.com/DataSlingers/ClusterLOCO/actions/workflows/release.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyPI version](https://badge.fury.io/py/clim.svg)](https://badge.fury.io/py/clim)

# Cluster LOCO: Feature Importance for Interpreting Clusters

`clim` is a Python package for Cluster LOCO feature importance methods for clustering interpretability.
Cluster LOCO is a model-agnostic framework for quantifying feature importance in clustering. The package provides methods that evaluate how much removing a feature affects the generalizability and stability of a clustering solution, enabling feature-level interpretation for unsupervised learning workflows.

## Installation

Clone the repository and install the package in editable mode:
```bash
git clone https://github.com/DataSlingers/ClusterLOCO.git
cd ClusterLOCO
pip install -e .
```

For experiment dependencies, such as anndata and scanpy, install:
```bash
pip install -e ".[experiments]"
```

### Requirements

The core package requires Python 3.10 or higher. Core dependencies include:
```bash
numpy
scipy
pandas
scikit-learn
joblib
tqdm
matplotlib
seaborn
leidenalg
igraph
```
Optional experiment dependencies include:
```bash
anndata
scanpy
```

## Get started

The package offers Cluster LOCO via data splitting, via minipatches and with adaptive recursive trimming. Two example notebooks for running our models are available under the `example` folder with simulated data and a real application to PBMC 68k data. For the latter you will need to install `anndata` and `scanpy`. 

### Cluster LOCO Split 
Cluster LOCO Split is recommended for data with *few features* (less than 10 features). 
```python
from clim.data_splitting import Cluster_LOCO_Split
from clim.utils import hinge_error
```
**Basic usage**: for any `sklearn` clustering algorithm, default transfer classifier is `RandomForestClassifier`. 
```python
from sklearn.cluster import SpectralClustering
model = SpectralClustering(n_clusters=K) 
feature_importance, feature_importance_se = Cluster_LOCO_Split(X_train, X_test, model=model, error_metric=hinge_error, use_proba=True, seed=42)
``` 

### Cluster LOCO-MP
Cluster LOCO-MP implements a minipatch ensemble version of Cluster LOCO. This approach is suited for large data. 

```python
from clim import ClusterLOCOMP
```
**Basic usage**: for any `sklearn` clustering algorithm, first `fit()` the minipatch model, then compute the feature importance via `score()`. We recommend to use parallelization during model fitting but **not** during computing scores where the overhead can be consequential. 
```python
g = ClusterLOCOMP(base_clusterer = model, base_classifier = RandomForestClassifier(), K=3, B=500)
g.fit(X, standardize=False, alpha_N = 0.2, alpha_M = 0.2, parallel=par)
out = g.score(error_metric=hinge_error, agg='mean', proba_error=True, parallel_features=False) 
```

### Cluster LOCO-RAMPART
Cluster LOCO-RAMPART is a sped-up version of Cluster LOCO-MP based on adaptive recursive trimming of active feature set. We recommend using this with high-dimensional data. 

**Basic usage**:
```python
from clim import ClusterLOCO_RAMPART, RAMPART
from clim.utils import transform_scores_to_ranking
```
`RAMPART` directly fits the model and computes the scores. 
```python
gen_fn = ClusterLOCO_RAMPART(base_clusterer=model, K=3, error_metric=hinge_error, parallel_MP=True,
    parallel={"n_jobs_features": 3, "backend": "loky", "prefer": "processes", "verbose": 0}, 
    standardize=False, alpha_N = 0.2, alpha_M = 0.2)
out = RAMPART(X, generalizability_fn=gen_fn, B=1000, ranking_fn=transform_scores_to_ranking, top_k=50)
```

## Other 
*Development install*

To install locally while developing:
```bash
pip install -e .
```
To check that the package is correctly installed:
```bash
python -c "import clim; print(clim.__file__)"
python -c "from clim import ClusterLOCOMP; print('import ok')"
```

To build the package:
```bash
python -m pip install build
python -m build
```
This should create a source distribution and wheel in the dist/directory.

*Optional experiment dependencies*:

The package keeps experiment dependencies separate from the core installation. This avoids requiring single-cell analysis packages for users who only want the core Cluster LOCO methods.

Install experiment dependencies with:
```bash
pip install -e ".[experiments]"
```
This installs additional packages such as:
```bash
anndata
scanpy
```

### Package structure
```
ClusterLOCO/
├── pyproject.toml
├── README.md
└── clim/
    ├── __init__.py
    ├── minipatches/
    ├── data_splitting/
    ├── models/
    └── utils/
└── benchmarking/
└── simulations/
└── example/
└── paper_figures/
```

This repository additionally contains code to reproduce the results from our paper: the folder `paper_figures` contains the notebook to make the main figures from our paper. 

## Citation

If you use this package, please cite the corresponding Cluster LOCO paper.
```
@preprint{he2026clusterloco,
  title={Cluster LOCO: Feature Importance for Interpreting Clusters},
  author={He, Claire and Allen, Genevera},
  url={https://arxiv.org/pdf/2606.14592},
  year={2026}
}
```
## License

MIT License
