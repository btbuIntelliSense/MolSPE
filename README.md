# MolSGPE: Unified Dynamic-static Pre-training Framework for Molecular Property Prediction

## Abstract
Molecular property prediction is a crucial foundation in food flavor optimization and drug discovery, facilitating the efficient screening of bioactive compounds with desirable properties. To obtain effective representations, deep-learning approaches via pre-training strategy are often employed to learn general features from large-scale unlabeled datasets. However, obtaining a more comprehensive molecular representation remains a challenge.

In this paper, a novel pre-trained modeling framework, named **MolSGPE** (Molecular Static Geometry and Potential Energy), is proposed to efficiently characterize molecules in multiple dimensions and their potential correlations. The MolSGPE achieves molecular representations through unifying static and dynamic features:
- **Static features** are learned via generating molecular geometries and employing mask reconstruction.
- **Dynamic features** are captured through perturbing equilibrium conformations and denoising.

Additionally, a **dual graph neural network** is constructed within the model to capture multi-granularity molecular features and their motif-based associative relationships, comprising both motif nodes and molecular nodes.

The proposed MolSGPE was evaluated across multiple molecular classification datasets. It outperformed state-of-the-art models in five out of six datasets. Notably, MolSGPE demonstrated exceptional performance in flavor recognition tasks, achieving a particular ROC-AUC of **0.896 for the sweet flavor**, highlighting its potential for industrial applications.

## Project Structure & File Descriptions

The roles of the main python program files are as follows:

1. **pretrain.py**: Main program for the pre-training stage. It handles the loading of large-scale unlabeled data and executes the static/dynamic feature learning tasks (mask reconstruction and denoising).
2. **count.py**: Likely used for statistical analysis of datasets or counting model parameters/results during the experimental phase.
3. **t-SNE.py**: Visualization script. It uses the t-SNE algorithm to project high-dimensional molecular representations into 2D space for clustering analysis (generating `clustering_result.png`).
4. **atom_types.py**: Utility script defining atom types and chemical element mappings required for graph construction.
5. **__init__.py**: Initialization file for the Python package structure.

*(Note: The folders `Datamodel`, `DeepCCA`, `EMPP`, `MaskModel`, `model`, and `mouels` contain the core architecture definitions, including the Dual Graph Neural Network and specific modules for geometry and potential energy processing.)*

## Dataset Description

The `dataset/` folder contains the benchmarks used for evaluation, covering drug discovery and food science domains:

### Molecular Property Prediction Benchmarks
These datasets are widely used to evaluate the ability of models to predict biological and chemical properties:

1.  **BACE**: Beta-secretase 1 binding affinity dataset. Used to predict whether a molecule inhibits the human beta-secretase 1 enzyme.
2.  **BBBP**: Blood-Brain Barrier Penetration dataset. A binary classification task to predict blood-brain barrier permeability.
3.  **ClinTox**: Clinical Toxicity dataset. Contains drugs approved by the FDA and those that have failed clinical trials due to toxicity.
4.  **Tox21**: Toxicology in the 21st Century dataset. Includes qualitative toxicity measurements for over 12,000 compounds on 12 different targets.
5.  **ToxCast**: Toxicology Forecaster dataset. A broad collection of in vitro high-throughput screening data for toxicity prediction.
6.  **SIDER**: Side Effect Resource dataset. Contains marketed drugs and their adverse drug reactions, grouped into system organ classes.
7.  **HIV**: HIV inhibition dataset. Experimental results measuring the ability of compounds to inhibit HIV replication.
8.  **MUV**: Maximum Unbiased Validation dataset. Designed specifically for benchmarking virtual screening methods with a focus on reducing false positives.

### Flavor & Specialized Datasets
9.  **flavor**: A specialized dataset focused on food science, used to evaluate the model's performance in recognizing specific taste profiles (e.g., Sweet, Bitter). This dataset highlights the model's industrial application potential in flavor optimization.

### Pre-training Data
10. **pretrain / pretrainall**: Large-scale unlabeled molecular datasets (likely derived from ZINC or similar libraries) used for the self-supervised pre-training phase to learn general molecular representations.
11. **test**: Held-out test sets used for final model evaluation.
