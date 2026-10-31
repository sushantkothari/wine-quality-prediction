
<div align="center">

# Wine Quality Prediction

### Machine Learning Classification Using Physicochemical Properties of Red Wine

<p>
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Scikit-learn"/>
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter"/>
</p>

<p>
  <img src="https://img.shields.io/badge/Domain-Machine%20Learning-0A9396?style=flat-square" alt="Machine Learning"/>
  <img src="https://img.shields.io/badge/Task-Binary%20Classification-7B2FBE?style=flat-square" alt="Binary Classification"/>
  <img src="https://img.shields.io/badge/Dataset-Red%20Wine%20Quality-blue?style=flat-square" alt="Red Wine Quality Dataset"/>
</p>

**A machine learning project that explores red wine quality classification using physicochemical measurements and compares multiple classification algorithms.**

</div>

---

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Models](#models)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Key Observations](#key-observations)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Running the Notebook](#running-the-notebook)
- [Technology Stack](#technology-stack)
- [Limitations and Future Improvements](#limitations-and-future-improvements)
- [Acknowledgements](#acknowledgements)
- [Author](#author)

---

## Project Overview

Wine quality is influenced by several physicochemical properties, including acidity, residual sugar, chlorides, sulphates, density, and alcohol content. This project applies machine learning techniques to explore the relationship between these properties and wine quality labels.

Multiple classification algorithms are trained and evaluated using a Jupyter notebook. Their performance is compared using accuracy, precision, recall, and F1-score to understand their effectiveness in predicting the target classes.

The repository contains the red wine dataset and the notebook used for data analysis, model training, and evaluation.

## Objectives

- Explore the red wine quality dataset and its physicochemical features.
- Prepare data for supervised machine learning.
- Train and compare multiple classification algorithms.
- Evaluate model performance using standard classification metrics.
- Analyze class-level performance and the impact of class imbalance.
- Document the machine learning workflow and evaluation results.

## Dataset

This project uses the **Red Wine Quality dataset** associated with the UCI Machine Learning Repository.

The dataset contains physicochemical measurements of red wine samples and their corresponding quality scores.

### Features

| Feature | Description |
|---|---|
| Fixed acidity | Concentration of non-volatile acids |
| Volatile acidity | Concentration of volatile acids |
| Citric acid | Citric acid content |
| Residual sugar | Sugar remaining after fermentation |
| Chlorides | Chloride concentration |
| Free sulfur dioxide | Free sulfur dioxide concentration |
| Total sulfur dioxide | Total sulfur dioxide concentration |
| Density | Density of the wine |
| pH | Measure of acidity or alkalinity |
| Sulphates | Sulphate content |
| Alcohol | Alcohol content by volume |
| Quality | Recorded wine quality score |

**Dataset file:** `winequality-red.csv`

The notebook uses classification labels derived from the quality data. Refer to the notebook for the exact target-label transformation and class definitions.

## Project Workflow

The project follows a supervised machine learning workflow:

1. **Data Loading:** Load the dataset into a Pandas DataFrame.
2. **Exploratory Data Analysis:** Examine feature distributions and relationships.
3. **Data Preprocessing:** Prepare input features and target labels.
4. **Data Splitting:** Separate data for model training and evaluation.
5. **Model Training:** Train multiple classification algorithms.
6. **Model Evaluation:** Generate classification reports and calculate evaluation metrics.
7. **Model Comparison:** Compare model performance across the target classes.

The notebook contains the implementation details for the preprocessing steps, data split, and model configurations.

## Models

The project evaluates the following classification algorithms:

| Model | Description |
|---|---|
| Logistic Regression | Linear classification algorithm that estimates class probabilities |
| Random Forest Classifier | Ensemble learning algorithm based on multiple decision trees |
| Support Vector Machine (SVM) | Classification algorithm that identifies a separating decision boundary |
| Gradient Boosting Classifier | Ensemble algorithm that sequentially improves predictions using decision trees |

Each model provides a different approach to learning patterns in the dataset. Their performance is compared using classification metrics.

## Evaluation Metrics

The models are evaluated using the following metrics:

- **Accuracy:** Proportion of correctly classified samples.
- **Precision:** Proportion of positive predictions that are correct.
- **Recall:** Proportion of actual positive samples correctly identified.
- **F1-Score:** Harmonic mean of precision and recall.
- **Support:** Number of actual samples belonging to each class.

Accuracy provides an overall measure of performance, while precision, recall, and F1-score help identify weaknesses in individual classes.

## Results

The following results are taken from the classification reports documented for the models.

The evaluation set contains **318 samples**, comprising 278 samples from class `0.0` and 40 samples from class `1.0`.

### Model Comparison

| Model | Accuracy | Class 1 Precision | Class 1 Recall | Class 1 F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.87 | 0.00 | 0.00 | 0.00 |
| Random Forest | 0.87 | 0.33 | 0.03 | 0.05 |
| Support Vector Machine | 0.87 | 0.00 | 0.00 | 0.00 |
| Gradient Boosting | 0.87 | 0.40 | 0.05 | 0.09 |

### Logistic Regression

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| 0.0 | 0.87 | 1.00 | 0.93 | 278 |
| 1.0 | 0.00 | 0.00 | 0.00 | 40 |
| Macro Average | 0.44 | 0.50 | 0.47 | 318 |
| Weighted Average | 0.76 | 0.87 | 0.82 | 318 |

**Accuracy:** 0.87

### Random Forest Classifier

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| 0.0 | 0.88 | 0.99 | 0.93 | 278 |
| 1.0 | 0.33 | 0.03 | 0.05 | 40 |
| Macro Average | 0.60 | 0.51 | 0.49 | 318 |
| Weighted Average | 0.81 | 0.87 | 0.82 | 318 |

**Accuracy:** 0.87

### Support Vector Machine (SVM)

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| 0.0 | 0.87 | 1.00 | 0.93 | 278 |
| 1.0 | 0.00 | 0.00 | 0.00 | 40 |
| Macro Average | 0.44 | 0.50 | 0.47 | 318 |
| Weighted Average | 0.76 | 0.87 | 0.82 | 318 |

**Accuracy:** 0.87

### Gradient Boosting Classifier

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| 0.0 | 0.88 | 0.99 | 0.93 | 278 |
| 1.0 | 0.40 | 0.05 | 0.09 | 40 |
| Macro Average | 0.64 | 0.52 | 0.51 | 318 |
| Weighted Average | 0.82 | 0.87 | 0.82 | 318 |

**Accuracy:** 0.87

## Key Observations

- All four models report an accuracy of approximately **87%**.
- The evaluation set has an imbalanced class distribution, with significantly fewer class `1.0` samples.
- Logistic Regression and SVM report zero recall for class `1.0`.
- Random Forest achieves a class `1.0` recall of `0.03`.
- Gradient Boosting achieves the highest reported class `1.0` precision and recall among the four models, at `0.40` and `0.05`, respectively.
- Despite similar accuracy scores, the models show limited ability to identify class `1.0` correctly.

These results demonstrate why accuracy should not be considered in isolation when evaluating classification models on imbalanced datasets.

## Repository Structure

```text
wine-quality-prediction/
├── README.md
├── wine-quality-prediction.ipynb
└── winequality-red.csv
```

| File | Description |
|---|---|
| `README.md` | Project documentation |
| `wine-quality-prediction.ipynb` | Notebook containing the analysis and model implementation |
| `winequality-red.csv` | Red wine quality dataset |

## Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook or JupyterLab
- Required Python data science and machine learning libraries

### 1. Clone the Repository

```bash
git clone https://github.com/sushantkothari/wine-quality-prediction.git
cd wine-quality-prediction
```

### 2. Create a Virtual Environment

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install jupyter pandas numpy matplotlib seaborn scikit-learn
```

If the notebook imports additional packages, install those dependencies as needed.

## Running the Notebook

Start Jupyter Notebook from the project directory:

```bash
jupyter notebook
```

Open `wine-quality-prediction.ipynb` and execute the cells sequentially.

The notebook provides the implementation for data preparation, classification model training, and performance evaluation.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Jupyter Notebook | Interactive development and experimentation |
| Pandas | Data loading and manipulation |
| NumPy | Numerical computations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning models and evaluation metrics |

## Limitations and Future Improvements

The reported results indicate that all four models struggle to identify class `1.0` reliably. Several improvements could be explored:

- **Class Imbalance:** Evaluate class weighting and resampling techniques using training data only.
- **Hyperparameter Tuning:** Optimize model parameters using cross-validation.
- **Additional Metrics:** Evaluate balanced accuracy, macro F1-score, and confusion matrices.
- **Data Leakage Prevention:** Fit preprocessing transformations only on training data.
- **Target Definition:** Document the precise mapping from original quality scores to classification labels.
- **Robust Validation:** Use stratified cross-validation and retain a separate test set for final evaluation.
- **Regression:** Explore regression models as a separate task to predict the original numeric wine quality score.

These are potential extensions and are not claims about functionality already implemented in the notebook.

## Acknowledgements

This project uses the Red Wine Quality dataset associated with the [UCI Machine Learning Repository](https://archive.ics.uci.edu/).

## Author

**Sushant Kothari**

- [GitHub Profile](https://github.com/sushantkothari)
- [Wine Quality Prediction Repository](https://github.com/sushantkothari/wine-quality-prediction)

---

<div align="center">

If you find this project useful, consider giving the repository a star.

</div>
