# MAIN-PROJECT-FAKE-NEWS-DETECTION
The dataset includes both reliable and misleading sources, making it suitable for NLP tasks, misinformation detection, and media bias analysis.
![](https://github.com/Nishathaj/MAIN-PROJECT-FAKE-NEWS-DETECTION/blob/main/Headerheader.jpg)

> 🔬 A data-driven machine learning project that predicts diabetes subtypes using genetic, environmental and lifestyle factors.

---

## 🧩 Project Overview

This project uses a comprehensive dataset encompassing multiple diabetes subtypes, including Steroid‑Induced Diabetes, Neonatal Diabetes Mellitus (NDM), Prediabetes, Type 1 Diabetes, and Wolfram Syndrome. The dataset includes medical, genetic, environmental, and lifestyle attributes to provide a holistic view of each patient’s profile. The aim is to help researchers and healthcare professionals understand how these factors interact and contribute to the development and progression of different diabetes subtypes, enabling insights for personalized treatment, risk assessment, and improved disease management.

### 🎯 Objectives
- Predict the diabetes subtype from patient health indicators.  
- Identify key predictors of each subtype.  
- Evaluate multiple models and select the best-performing classifier.  

---

### ▶️ Quick Start
1. Open the notebook:
### ▶️ Run Directly in Google Colab
You can execute the entire workflow without any setup:
🔗 [**Open Project in Colab**](https://colab.research.google.com/drive/1q-nUQ2FST6eJ09AvpHbYYH9nQqWF_7W7?usp=sharing)
#### Codes and Resources Used
- **Editor Used:** Google Colab / Jupyter Notebook  
- **Python Version:** 3.12  
- **Platform:** Google Colab  
- **Environment:** Machine Learning / Health Informatics  

#### Python Packages Used
- **General Purpose:** `os`, `warnings`, `joblib`, `requests`  
- **Data Manipulation:** `pandas`, `numpy`  
- **Data Visualization:** `matplotlib`, `seaborn`, `plotly`  
- **Machine Learning:** `scikit-learn`, `xgboost`

# Data
The dataset is a crucial part of this project. It combines clinical, genetic, environmental, and lifestyle features to predict diabetes subtypes.

I structure this as follows - 

## Source Data
**Description:** Contains 70,000 patient samples with 34 features including genetic markers, autoantibodies, family history, environmental factors, lifestyle attributes, and clinical measures.

**Target Feature:** Diabetes_Subtype with 13 classes:
0: Steroid-Induced Diabetes
1: Neonatal Diabetes Mellitus (NDM)
2: Prediabetic
3: Type 1 Diabetes
4: Wolfram Syndrome
5: LADA
6: Type 2 Diabetes
7: Wolcott-Rallison Syndrome
8: Secondary Diabetes
9: Type 3c Diabetes
10: Gestational Diabetes
11: Cystic Fibrosis‑Related Diabetes (CFRD)
12: MODY

## Data Acquisition
- Data can be downloaded directly from the repository or other open-source sources.

- In some cases, data may be collected via API calls or web scraping (elaborate if applicable).

- Ensure all license restrictions and credits are properly followed.
## Data Preprocessing
To make the dataset suitable for modeling:

1.Checked for missing values → none found

2.Verified duplicate rows → none found

3.Removed outliers using IQR method

4.Applied skewness correction and transformations (log/square root)

5.Scaled numeric features using StandardScaler / MinMaxScaler

6.Encoded categorical variables using one-hot encoding or label encoding

## 📊📊 Workflow Steps

| Step                       | Description                                             |
| -------------------------- | ------------------------------------------------------- |
| 1️⃣ Load Dataset           | Import raw data into environment                        |
| 2️⃣ Initial EDA            | Analyze distributions, missing values, outliers         |
| 3️⃣ Data Preprocessing     | Handle nulls, outliers, encode categorical features     |
| 4️⃣ Feature Engineering    | Create new features or transform existing ones          |
| 5️⃣ Feature Scaling        | Standardize or normalize numeric features               |
| 6️⃣ Feature Selection      | Select top features using SelectKBest                   |
| 7️⃣ Train/Test Split       | Divide dataset into training and testing sets           |
| 8️⃣ Model Building         | Train multiple machine learning models                  |
| 9️⃣ Hyperparameter Tuning  | Optimize model parameters using RandomizedSearchCV      |
| 🔟 Model Evaluation        | Evaluate using Accuracy, F1, Precision, Recall, AUC-ROC |
| 1️⃣1️⃣ Final Prediction    | Predict diabetes subtype for new patient samples        |
| 1️⃣2️⃣ Future Enhancements | Deep learning, ensemble methods, deployment             |



#### 📊 Dataset
- **Rows × Columns:** Rows × Columns: 70,000 × 34

- **Features include:** genetic markers, autoantibodies, family history, environmental factors, lifestyle attributes, clinical measures

- **No missing values or duplicates**

#### 🤖 Model Building
#### 🧩 Algorithms Used

The following machine learning algorithms were implemented and compared to identify the best-performing model for multiclass diabetes subtype prediction:

- Random Forest Classifier 🌲

- Logistic Regression 📈

- Naive Bayes Classifier 🧮

- Decision Tree Classifier 🌳

- Gradient Boosting Classifier 🚀

- Support Vector Machine (SVM) ⚙️

- K-Nearest Neighbors (KNN) 👥

#### 🎯 Model Tuning

**Hyperparameter Optimization:** Conducted using GirdsearchSearchCV

**Parameters fine-tuned include:**

- Number of estimators

- Maximum depth

- Learning rate (for boosting models)

- Regularization parameters

- Kernel type and C value (for SVM)

#### 🧠 Model Evaluation Metrics

Each model was evaluated using multiple metrics to ensure balanced performance across all diabetes subtypes:

#### Metric	Description
| Metric                   | Description                                             |
| :----------------------- | :------------------------------------------------------ |
| **Accuracy**             | Overall proportion of correctly classified samples      |
| **Precision**            | Fraction of correctly predicted positive observations   |
| **Recall (Sensitivity)** | Fraction of actual positives correctly identified       |
| **F1-Score**             | Harmonic mean of Precision and Recall                   |
| **AUC-ROC**              | Measures model's ability to distinguish between classes |

#### 🏆 Best Model

#### After comprehensive evaluation:

#### Best Performing Model: 🏅 Random Forest Classifier

#### Reason for Selection:

- Highest accuracy and AUC-ROC scores

- Well-balanced precision–recall trade-off

- High interpretability through feature importance

- Robust performance against noise and feature correlations

#### 📈 Sample Prediction
#### Sample Input

| Feature              | Value |
| -------------------- | ----- |
| Age                  | 45    |
| Blood Pressure       | 130   |
| Blood Glucose Levels | 210   |
| BMI                  | 29.3  |
| Genetic Marker 1     | 1     |
| Family History       | Yes   |
| Physical Activity    | Low   |
| Diet Score           | 6     |

#### Sample Output

| Diabetes Subtype         | Probability |
| ------------------------ | ----------- |
| Prediabetes              | 0.65        |
| Type 1 Diabetes          | 0.15        |
| Steroid-Induced Diabetes | 0.05        |
| Wolfram Syndrome         | 0.02        |
| NDM                      | 0.13        |

#### Predicted Subtype: Prediabetes

**Top Contributing Features:** Genetic Marker 1, BMI, Diet Score

## Final Conclusion

1.A comprehensive machine learning pipeline was developed to predict diabetes subtypes using clinical, genetic, environmental, and lifestyle factors.

2.Multiple models were trained and evaluated: Random Forest, Logistic Regression, Naive Bayes, Decision Tree, Gradient Boosting, SVM, and KNN.

3.Random Forest emerged as the best-performing model, achieving 92% accuracy and high performance across Precision, Recall, F1-Score, and AUC-ROC metrics.

4.Key predictive features include: BMI, Blood Glucose Levels, Genetic Marker 1, Diet Score, Insulin Levels, and Family History.

5.The model effectively captures complex interactions between genetic, lifestyle, and clinical factors, providing interpretable insights for diabetes subtype prediction.

6.This project demonstrates the potential of machine learning in healthcare analytics for early detection, personalized risk assessment, and targeted interventions.

7.Future work can include deep learning models, ensemble methods, explainable AI (SHAP/LIME), and deployment as a web/mobile application.

# 🚀 Future Enhancements
Outline potential future work that can be done to extend the project or improve its functionality. This will help others understand the scope of your project and identify areas where they can contribute.
**1.Hyperparameter Optimization:** Use finer GridSearch or Bayesian optimization.

**2.Feature Engineering:** Add derived features, feature selection, or PCA.

**3.Class Imbalance Handling:** Use SMOTE or class weighting.

**4.Ensemble Learning:** Explore stacking or boosting (XGBoost, LightGBM).

**5.Data Expansion:** Incorporate additional genetic/environmental/lifestyle factors.

**6.Deployment & Monitoring:** Real-time prediction pipeline with continuous monitoring and retraining.

# Model Optimization

**1.Address Class Imbalance**

- Apply SMOTE, class weighting, or targeted oversampling/undersampling to improve predictions for underrepresented classes.

**2.Ensemble Techniques**

- Combine your trained models (Random Forest, Logistic Regression, Naive Bayes) using stacking to leverage complementary strengths.

- Explore boosting models like XGBoost, LightGBM, or CatBoost for higher predictive performance.

**3.Cross-Validation Enhancements**

- Use stratified k-fold cross-validation to ensure consistent performance across all classes.

- Monitor metrics per fold to detect potential overfitting

# Acknowledgments/References
Acknowledge any contributors, data sources, or other relevant parties who have contributed to the project. This is an excellent way to show your appreciation for those who have helped you along the way.
- Dataset inspired by open-source health data repositories.

- README template adapted from Pragyy’s Data Science Readme Template


# License
Specify the license under which your code is released. Moreover, provide the licenses associated with the dataset you are using. This is important for others to know if they want to use or contribute to your project. 

For this github repository, the License used is [MIT License](https://opensource.org/license/mit/).
