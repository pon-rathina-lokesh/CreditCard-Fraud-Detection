# Credit Card Fraud Detection


📌 Overview

Credit card fraud is a significant problem in the financial industry.  
This project aims to detect fraudulent credit card transactions using Machine Learning techniques.  
We use Random Forest Classifier to build a reliable fraud detection model.

---

📂 Dataset

-Source: [Credit Card Fraud Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023?select=creditcard_2023.csv)

Description:  
   This dataset contains credit card transactions made by European cardholders in the year 2023. It comprises over 550,000 records, and the data
   has been anonymized to protect the cardholders' identities. The primary objective of this dataset is to facilitate the development of fraud detection
   algorithms and models to identify potentially fraudulent transactions.

Features:  
  -V1 to V28: PCA transformed features (sensitive data hidden)
  - Time & Amount: Transaction details
  - Class: Target variable (0 → Normal, 1 → Fraudulent)

---

 🔍 Project Pipeline

1. Exploratory Data Analysis (EDA):
   - Visualized class imbalance.
   - Checked correlations between features.
   - Plotted fraud vs. non-fraud transaction patterns.

2. Data Preprocessing:
   - Handled missing values (if any).
   - Scaled features using `StandardScaler`.

3. Model Training:
   - Used Random Forest Classifier for classification.
   - Tuned hyperparameters for better accuracy.

4. Evaluation:
   - Confusion Matrix
   - ROC-AUC Score
   - Precision, Recall, F1-score

---

Results
- Accuracy: ~99%
- AUC-ROC Score: ~0.99
- The model successfully detects fraudulent transactions with high recall.


---


⚙️ How to Run This Project

🚀 How to Run This Project on Google Colab

You can run this project directly on Google Colab without installing anything locally.

Steps:

1. Open the Colab Notebook
   - [Click here to open the notebook in Google Colab](https://colab.research.google.com/drive/1SpYu-B0DzVXg4kIL9ktzUbMyDAK3V8yi?usp=sharing)

2. Upload the Dataset  
   - If you have the dataset locally, upload it by running:
     ```python
     from google.colab import files
     uploaded = files.upload()
     ```
   - Or download directly from Kaggle:
     

3. Install Required Libraries (if needed)  
   - Colab already includes most libraries like `pandas`, `numpy`, and `scikit-learn`.  
     If any library is missing, install it.


4. Run All Cells  
   - Click on Runtime → Run all to execute the entire pipeline (EDA → Model Training → Evaluation).

---

Outputs Provided
- Confusion Matrix
- ROC Curve
- Feature Importance Visualizations

---


🛠️ Technologies Used
- Python (Google Colab)
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn


