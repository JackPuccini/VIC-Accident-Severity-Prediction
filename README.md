# COMP20008 Assignment 2 - Group W16G7

This research project addresses the question "Can accident severity be predicted using machine learning models?" using datasets from the Victoria Road Crash dataset.

---

## Table of Contents

1. [Setup](#setup)  
2. [Data](#data)  
3. [Preprocessing](#preprocessing)  
4. [Clustering Analysis](#clustering-analysis)  
5. [EDA and Feature Analysis](#eda)  
6. [Model Training](#model-training)  
7. [Running Code](#running-code)
8. [Intended Workflow](#intended-workflow)
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

## Running Code

All .py scripts can be run directly. Jupyter notebooks should be executed from start to finish.

## Recommended Workflow

The recommended workflow is to:

1. Preprocess all raw data to generate preprocessed versions.  
2. Generate the final integrated dataset.  
3. Conduct clustering analysis which updates the integrated dataset prior to feature analysis and model training.  
4. Conduct feature correlation and exploratory data analysis which outputs the `top_10_train.csv` file containing only the top 10 features.  
5. Conduct machine learning model evaluations using both the full dataset and the truncated, top 10 feature dataset.  

