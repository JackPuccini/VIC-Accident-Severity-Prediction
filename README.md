# COMP20008 Assignment 2 - Group W16G7

This research project addresses the question "Can accident severity be predicted using machine learning models?" using datasets from the Victoria Road Crash dataset.

---

## Table of Contents

1. [Setup](#setup)  
2. [Data](#data)  
3. [Preprocessing](#preprocessing)  
4. [Clustering Analysis](#clustering-analysis)  
5. [EDA and Feature Analysis](#eda-and-feature-analysis)  
6. [Model Training](#model-training)  
7. [Running Code and Intended Workflow](#running-code-and-intended-workflow)
8. [Acknowledgements](#acknowledgements)
---

## Setup
1. Create and activate a virtual env:
   ```bash
   python3 -m venv venv
   venv/Scripts/activate  
2. Install dependencies
    ```bash 
    pip install -r requirements.txt
## Data
All datasets (raw and processed) are present in the data directory:
```
📦data
┣ 📂datasets
┣ 📂processed
┗ 📂train_test
```

## Proprocessing
Scripts under src/preprocessing preprocess the raw datasets, ultimately generating the `integrated_data.csv` file.
```
📦preprocessing
┣ 📂accident
┣ 📂data_integration
┣ 📂person
┣ 📂surface_and_atmosphere
┗ 📂vehicle
```

## Clustering Analysis
Notebook under src/clustering performs the clustering analysis.
```
📦clustering
┗ 📜clustering_analysis.ipynb
```

## EDA and Feature Analysis
Notebook under src/eda performs the feature correlation analysis as well as exploratory data analysis.
```
📦eda
┗ 📜eda_feature_analysis.ipynb
```

## Model Training
Two notebooks under src/model_training:
```
📦model_training
 ┣ 📜all_features_test_suite.ipynb
 ┗ 📜top_10_test_suite.ipynb
```
Each trains and evaluates models (Decision Tree, Random Forest, K-NN) on the respective feature sets.

## Running Code and Intended Workflow

All .py scripts can be run directly. Jupyter notebooks should be executed from start to finish.

The recommended workflow is to:

1. Preprocess all raw data to generate preprocessed versions (run `accident_preprocessing.py`, `person_preprocessing.py`, `surface_and_atmosphere_preprocessing.py` and`vehicle_preprocessing.py`).
2. Generate the final integrated dataset (run `data_integration.py`).  
3. Conduct clustering analysis which updates the integrated dataset prior to feature analysis and model training (execute the `clustering_analysis.ipynb` notebook).  
4. Conduct feature correlation and exploratory data analysis which outputs the `top_10_train.csv` dataset containing only the top 10 features (execute the `eda_feature_analysis.ipynb` notebook).  
5. Conduct machine learning model evaluations using both the full dataset and the truncated, top 10 feature dataset (execute the `all_features_test_suite.ipynb` and `top_10_test_suite.ipynb` notebooks).


## Acknowledgements

We acknowledge that generative AI tools were used to support code development for this report. These tools were employed to assist in understanding the availability and correct usage of functions from Python libraries such as scikit-learn, matplotlib, and pandas.
