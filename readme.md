

---

# **Breast Cancer Diagnosis: Feature Importance and Model Comparison**

## **Project Overview**
This project analyzes the Breast Cancer Wisconsin (Diagnostic) dataset to classify tumors as benign or malignant using machine learning models. The primary objectives are:
1. To identify the most important features for predicting tumor malignancy.
2. To compare the performance of three machine learning models:
   - Lasso Regression (L1 Regularization)
   - Ridge Regression (L2 Regularization)
   - Random Forest Classifier

The analysis includes feature importance visualization, PCA-based data visualization, and model evaluation metrics. This project demonstrates machine learning techniques for feature selection, model evaluation, and interpretability.

---

## **Dataset**
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- **Description**: 
  - The dataset contains 569 samples of tumor measurements.
  - Each sample includes 30 numerical features derived from cell nuclei measurements.
  - Target variable: Diagnosis (`M` for Malignant, `B` for Benign).

### **Feature Information**
- **Numerical Features**:
  - Examples: `radius_mean`, `texture_mean`, `perimeter_mean`, etc.
- **Target Variable**:
  - Malignant (`M`) → 1
  - Benign (`B`) → 0

---

## **Methods and Workflow**

### **1. Data Preprocessing**
- **Steps**:
  - Features were standardized using `StandardScaler`.
  - Target variable was encoded as binary (Malignant = 1, Benign = 0).
  - The dataset was split into training (80%) and testing (20%) sets.
  
### **2. Models Trained**
- **Lasso Regression (L1)**:
  - Identifies key predictive features by shrinking irrelevant feature coefficients to zero.
- **Ridge Regression (L2)**:
  - Retains all features but penalizes large coefficients to prevent overfitting.
- **Random Forest Classifier**:
  - Uses ensemble learning to estimate feature importance and improve accuracy.

### **3. Feature Importance Analysis**
- Feature importance was evaluated using:
  - Coefficients from Lasso and Ridge Regression.
  - Feature importance scores from Random Forest.

### **4. Visualization**
- **PCA (Principal Component Analysis)**:
  - Reduced the dataset to two dimensions for visualization.
  - Highlighted class separability between benign and malignant samples.
- **Feature Importance Bar Plot**:
  - Compared importance scores across Lasso, Ridge, and Random Forest models.

---

## **Results**

### **1. Model Performance**
| Metric              | Lasso Regression | Ridge Regression | Random Forest |
|---------------------|------------------|------------------|---------------|
| **Accuracy**        | 96%             | 99%             | 96%          |
| **Precision (Malignant)** | 95%             | 100%            | 98%          |
| **Recall (Malignant)**    | 95%             | 98%             | 93%          |
| **F1-Score (Malignant)**  | 95%             | 99%             | 95%          |

- Ridge Regression emerged as the best-performing model with 99% accuracy and near-perfect precision and recall.

### **2. Key Features**
- **Top Features by Importance**:
  - `worst radius`, `mean texture`, and `worst smoothness` were consistently ranked as critical predictors across all models.

---

## **Conclusion**
- **Ridge Regression** is the most effective model for this dataset, achieving near-perfect classification performance.
- Lasso Regression provides a sparse, interpretable model, making it suitable for feature selection.
- Random Forest offers a strong baseline and interpretable feature importance scores.

This analysis demonstrates the power of regularization techniques and ensemble methods in medical diagnostics. The PCA visualization highlights the separability of the dataset, reinforcing the reliability of the models.

---

## **Project Structure**
```
|-- README.md
|-- feature_importances.csv     # Feature importance scores
|-- breast_cancer_analysis.py   # Python script for analysis
|-- figures/
    |-- pca_visualization.png   # PCA scatter plot
    |-- feature_importance.png  # Feature importance bar plot
```

---

## **Technologies Used**
- **Programming Language**: Python
- **Libraries**:
  - Data Manipulation: `pandas`, `numpy`
  - Machine Learning: `scikit-learn`
  - Visualization: `matplotlib`

---

## **How to see results**

    Click on breast_cancer_diagnosis.ipynb

---

## **Future Work**
- Hyperparameter tuning for Random Forest to improve recall.
- Expanding the analysis with additional models such as Support Vector Machines (SVM) or Gradient Boosting.
- Deploying the best model as a web application using Flask or Streamlit.

---

## **Acknowledgments**
- UCI Machine Learning Repository for the dataset.
- scikit-learn for providing versatile machine learning tools.

---

## **Contact**
For questions or collaborations, reach out at:
- **Email**: puigdedios@gmail.com

