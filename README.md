# MAIN-PROJECT-FAKE-NEWS-DETECTION
The dataset includes both reliable and misleading sources, making it suitable for NLP tasks, misinformation detection, and media bias analysis.
![](https://github.com/Nishathaj/MAIN-PROJECT-FAKE-NEWS-DETECTION/blob/main/Headerheader.jpg)

> 🔬 A data-driven machine learning project that predicts Fake news detection based on their textual content and source of charecterstics.

---

## 🧩 Project Overview

This project aims to build a machine learning model for fake news detection. So far,loaded and explored the dataset, performed extensive EDA, handled missing values and duplicates, renamed columns, encoded categorical features, extracted text features using TF-IDF, split the data, scaled numerical features, trained and evaluated multiple models, performed hyperparameter tuning, and analyzed feature importance. The project seems to have successfully achieved perfect accuracy on the dataset. The primary goal of this project is to build a machine learning model capable of automatically identifying and classifying news articles as either real or fake based on their textual content. This helps mitigate the spread of misinformation and supports informed public discourse.


### 🎯 Objectives

Fake news spreads fast and misleads millions. Manual fact-checking can't keep up. This project uses machine learning to automatically detect misinformation, helping protect public trust and promote reliable information online.This is in response to the rapid spread of fake news that manual fact-checking cannot keep up with.


---

### ▶️ Quick Start
1. Open the notebook:
### ▶️ Run Directly in Google Colab
You can execute the entire workflow without any setup:
🔗 [**Open Project in Colab**](https://colab.research.google.com/drive/1q-nUQ2FST6eJ09AvpHbYYH9nQqWF_7W7#scrollTo=7fVHsKIjzSBU)
#### Codes and Resources Used
- **Editor Used:** Google Colab / Jupyter Notebook  
- **Python Version:** 3.12  
- **Platform:** Google Colab  
- **Environment:** Machine Learning/ Classificatin Fake/Real

#### Python Packages Used
- **General Purpose:** `os`, `warnings`, `joblib`, `requests`  
- **Data Manipulation:** `pandas`, `numpy`  
- **Data Visualization:** `matplotlib`, `seaborn`, `plotly`  
- **Machine Learning:** `scikit-learn`, `xgboost`

# Data
The dataset is a crucial part of this project.It is designed for machine learning models to classify news as Real or Fake based on various linguistic, statistical, and factual parameters..

I structure this as follows - 

## Source Data
**Description:** Contains 4000 text samples with 24 features including with detailed metadata about news articles, including title, author, state of origin, sentiment score, source credibility, and more.   

**Target Feature:** LABEL.The target feature of this project is the Label column, which indicates whether a news article is 'Real' or 'Fake'. The goal is to predict this categorical outcome.

## Data Acquisition
- Data can be downloaded directly from the repository or kaggle sources.

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
- **Rows × Columns:** Rows × Columns: 4000× 24

- **Features include:**
  - Ctegorical Columns: Title, Author, Text, State, Date_published, Source, Category, Political_bias, Fact_check_rating, and Label (the target variable).
  - Numerical Columns: Id, Sentiment_score, Word_count, Char_count, Has_images, Has_videos, Readability_score, Num_shares, Num_comments, Is_satirical, Trust_score, Source_reputation, Clickbait_score, and Plagiarism_score.

- **No missing values or duplicates**

#### 🤖 Model Building
#### 🧩 Algorithms Used

The following machine learning algorithms were implemented and compared to identify the best-performing model:

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

Each model was evaluated using multiple metrics to ensure balanced performance across all textual contents:

#### Metric	Description
| Metric                   | Description                                             |
| :----------------------- | :------------------------------------------------------ |
| **Accuracy**             | Overall proportion of correctly classified Label   |
| **Precision**            | Fraction of correctly predicted positive observations   |
| **Recall (Sensitivity)** | Fraction of actual positives correctly identified       |
| **F1-Score**             | Harmonic mean of Precision and Recall                   |
| **AUC-ROC**              | Measures model's ability to distinguish between classes |

#### 🏆 Best Model

#### After comprehensive evaluation:

#### Best Performing Model: 🏅 XGBoost

#### Reason for Selection:

- Highest accuracy and AUC-ROC scores

- Well-balanced precision–recall trade-off

- High interpretability through feature importance

- Robust performance against noise and feature correlations

#### 📈 Sample Prediction

#### Sample Input                                                         |                                                            SAMPLE OUTPUT


 Text	                  | Source	   | Category	   | Political_bias	|Fact_check_rating	 | Sentiment_score	 |  Wrd_count	|  Char_count	  |Label
      

1541. It contai...      | Snopes	   | Sports	     | Center	         | Mixed	              |  0.92	         |     1497	   |    1354	    |   Fake

1982	This is the 
content of article 1983. 
It contai...	          | BBC	       | Technology  |  Left	         |   TRUE	              |  0.84	         |     594	   |    4943	    |   Fake

1718	This is the 
content of article 1719. 
It contai...	          | CNN	       | Sports	    |  Left	           |  FALSE	              |  -0.37	       |      224	   |    5802	    |   Real

3165	This is the 
content of article 3166. 
It contai...	          | Reuters	   | Business	  |  Right           |  Mixed	              | 0.47	         |     1500	   |    7501	    |   Real

358	This is the 
content of article 359. 
It contain...	          | The Onion  | Business	  |  Center	         |  TRUE	              | -0.67	         |    844	     |    867	      |   Fake










**Top Contributing Features:** Text, Source, Category	, Political_bias, Fact_check_rating, Sentiment_score, Wrd_count, Char_count

## Final Conclusion

1. All models achieved 100% accuracy and F1-score on the dataset.

2. No data leakage or overlap was detected between training and test sets.

3. TF-IDF + categorical + numeric features provided highly separable patterns.

4. XGBoost and Random Forest showed the highest feature importance values for trust_score, clickbait_score, and text TF-IDF tokens.

5. SVM and Logistic Regression performed equally well on the text representation.

6. The system can be integrated into real-time fake news detection pipelines or dashboards.










# 🚀 Future Enhancements
Outline potential future work that can be done to extend the project or improve its functionality. This will help others understand the scope of your project and identify areas where they can contribute.
**1.Hyperparameter Optimization:** Use finer GridSearch.

**2.Feature Engineering:** Add derived features, feature selection, or PCA.

**3.Class Imbalance Handling:** Use SMOTE or class weighting.

**4.Ensemble Learning:** Explore stacking or boosting (XGBoost, LightGBM).

**5.Data Expansion:** Incorporate additional Fake/Real News Factors.

**6.Deployment & Monitoring:** Real-time prediction pipeline with continuous monitoring and retraining.

# Model Optimization

**1.Address Class Imbalance**

- Apply SMOTE, class weighting, or targeted over sample text articles to improve predictions for underrepresented classes.

**2.Ensemble Techniques**

- Combine your trained models (Random Forest, Logistic Regression, Naive Bayes) using stacking to leverage complementary strengths.

- Explore boosting models like XGBoost, LightGBM for higher predictive performance.

**3.Cross-Validation Enhancements**

- Use stratified k-fold cross-validation to ensure consistent performance across all classes.

- Monitor metrics per fold to detect potential overfitting

# Acknowledgments/References
Acknowledge any contributors, data sources, or other relevant parties who have contributed to the project. This is an excellent way to show your appreciation for those who have helped you along the way.
- Dataset inspired by kaggle data repositories.

- README template adapted from Pragyy’s Data Science Readme Template


# License
Specify the license under which your code is released. Moreover, provide the licenses associated with the dataset you are using. This is important for others to know if they want to use or contribute to your project. 

For this github repository, the License used is [MIT License](https://opensource.org/license/mit/).
