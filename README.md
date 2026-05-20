# BioActivityML: Drug Activity Classification Comparative Study

## Overview

BioActivityML is a machine learning pipeline designed to predict whether a chemical compound is biologically active (1) or inactive (0) based on high-dimensional binary molecular descriptors.

Each compound is represented as a sparse binary feature vector encoding structural and chemical properties relevant to drug-receptor binding behavior.

The goal of this project is not only prediction, but a **systematic comparison of classical and modern machine learning approaches** under imbalanced, high-dimensional conditions.

---

## Problem Context

Drug discovery often begins with identifying whether small molecules bind to a target receptor. This process generates extremely high-dimensional datasets where:

- Each molecule is represented by thousands of binary features
- Only a small subset of compounds are actually active
- The dataset is highly imbalanced

In this project:
- 800 compounds are provided for training
- 350 compounds are used for testing (labels hidden)

Class distribution:
- ~78 active compounds (1)
- ~722 inactive compounds (0)

This imbalance makes standard accuracy insufficient, so **F1-score (macro)** is used as the primary evaluation metric.

---

## Objectives

The project explores four core machine learning models:

- k-Nearest Neighbors (k-NN)
- Naïve Bayes
- Decision Tree
- Neural Network (MLP)

The key focus is to:
- Compare model performance under identical data conditions
- Optimize each model independently using cross-validation
- Study the impact of feature selection and dimensionality reduction
- Handle severe class imbalance effectively

---

## Data Representation

Each molecule is encoded in a sparse binary format:
[0, 0, 1, 1, 0, 0, 1, 1, 1, 0] → [2, 3, 6, 7, 8]


Each line contains:
- Class label (training only)
- Indices of active features

This sparse representation reflects real-world chemical fingerprint datasets.

---

## System Pipeline

### 1. Data Loading
- Parse sparse binary input format
- Convert to compressed vector representations

### 2. Feature Engineering
- Sparse-to-dense transformation (when required)
- Feature filtering (low-variance removal)
- Dimensionality reduction techniques:
  - PCA (for dense models)
  - Feature selection via mutual information / chi-square

### 3. Model Training

Four classifiers are evaluated:

#### 🔹 k-NN
- Distance metrics:
  - Euclidean
  - Manhattan
  - Cosine similarity
- Hyperparameter: K selection via cross-validation

#### 🔹 Naïve Bayes
- Bernoulli Naïve Bayes for binary features
- Smoothing parameter tuning (alpha)

#### 🔹 Decision Tree
- Criteria comparison:
  - Gini vs Entropy
- Depth and pruning optimization

#### 🔹 Neural Network (MLP)
- Hidden layer tuning
- Activation function comparison
- Regularization (alpha) tuning

---

## Handling Class Imbalance

Due to extreme imbalance, multiple strategies were explored:

- Class weighting
- Threshold tuning
- Stratified cross-validation
- F1-score optimization instead of accuracy

These adjustments significantly improved minority class detection.

---

## Experimental Design

Each model was evaluated using:

- Stratified k-fold cross-validation
- Grid search over hyperparameters
- Consistent F1-macro evaluation metric

---

## Key Findings

### Model Performance Insights

- k-NN performed strongly with cosine similarity in sparse space
- Decision Trees benefited from depth control and feature pruning
- Naïve Bayes was stable but sensitive to feature correlation
- Neural Networks achieved best performance after careful tuning but required regularization to avoid overfitting

### Feature Engineering Impact

- Feature selection significantly improved all models
- Removing noisy/rare features improved generalization
- Dimensionality reduction helped neural networks the most

### Imbalance Effects

Without correction:
- Models biased heavily toward inactive class

With F1 optimization:
- Substantially improved recall for active compounds

---

## Performance Optimization

To handle computational and dimensional challenges:

- Sparse matrix representations were used throughout
- Feature pruning reduced noise and runtime
- Vectorized operations improved training efficiency
- Early stopping used for neural networks

---

## Tech Stack

- Python
- NumPy
- SciPy (sparse matrices)
- scikit-learn
- PyTorch / TensorFlow (optional MLP experiments)
- Jupyter Notebook

---

## Key Takeaways

This project demonstrates how model performance in biomedical classification tasks depends heavily on:

- Feature representation quality
- Proper handling of class imbalance
- Model-specific tuning strategies
- Evaluation metrics aligned with real-world objectives (F1 vs accuracy)

Rather than a single “best model,” the results highlight that **different models excel under different feature and tuning conditions**.

---

## Future Work

- Ensemble methods (stacking k-NN + MLP + Trees)
- SMOTE or synthetic oversampling techniques
- Graph-based molecular representations (GNNs)
- Automated hyperparameter optimization (Bayesian search)


Example:
