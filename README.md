
# Mushroom Classification: Edible vs Poisonous

## Overview  
This project uses machine learning to classify mushrooms as **edible** or **poisonous** based on 22 categorical features describing physical characteristics. The dataset originates from the **Audubon Society Field Guide to North American Mushrooms** and was donated to the UCI Machine Learning Repository in 1987. Given the real-life consequences of misidentification, this problem is both challenging and impactful.

## Problem Statement  
**Objective:**  
Build a classification model to predict whether a mushroom is **edible (`e`)** or **poisonous (`p`)** based on observable traits like odor, gill size, cap shape, and spore print color.

## Dataset

- **Samples:** 8,124
- **Features:** 22 categorical features + 1 target label (`class`)
- **Target Values:**
  - `e` = Edible
  - `p` = Poisonous  
  Note: The original "unknown edibility" category was combined with poisonous to ensure safety.
- **Source:** [https://www.kaggle.com/datasets/uciml/mushroom-classification)

## Steps Followed

### 1. Data Loading & Exploration
- Loaded and inspected the dataset using `pandas`
- Counted total rows and columns
- Identified that **all features are categorical**
- Noted missing values: `stalk-root` had ~30.5% missing, marked as `"?"`

### 2. Target & Class Distribution
- Verified target class (`class`) is nearly balanced
  - Edible: 51.8%
  - Poisonous: 48.2%
- Plotted distribution for transparency

### 3. Feature Analysis
- Used `pd.crosstab` to find features highly correlated with the poisonous class
- Top predictors included:
  - `odor`
  - `gill-size`
  - `spore-print-color`
- Created bar plots and histograms to visualize feature separability

### 4. Data Cleaning
- Dropped `veil-type` (constant feature)
- Replaced missing `stalk-root` values with its **mode**
- One-hot encoded all categorical features using `sklearn.OneHotEncoder` with `drop='first'`
- Final shape: **8124 samples × 92 encoded features**

### 5. Data Splitting
- Split into:
  - 80% training
  - 10% validation
  - 10% testing
- Used **stratified sampling** to maintain class balance

### 6. Model Training
- Chose **Random Forest Classifier**:
  - 100 trees
  - Class weight = "balanced"
  - Random seed for reproducibility

### 7. Evaluation Metrics
- Evaluated on both **validation** and **test** sets
- Reported:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC AUC
- Confusion matrices visualized using `seaborn`

### 8. Feature Importance
- Extracted top 20 features by importance from the trained model
- Displayed using a horizontal bar chart

## Project Structure

```bash
├── FinalProj.ipynb           # Main notebook with all steps documented
├── mushrooms.csv             # Original dataset
├── README.md                 # Project documentation

