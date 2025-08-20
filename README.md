# 🧠 NLP Emotion Classification  

## 📌 Overview  
This project focuses on **classifying emotions from text** using Natural Language Processing (NLP) and Machine Learning.  
The dataset consists of sentences labeled with emotions (e.g., joy, sadness, anger, etc.).  
Our goal is to build different models that can accurately detect emotions from raw text input.  

---

## ⚙️ Tech Stack  
- **Programming Language:** Python  
- **Libraries & Tools:**  
  - `pandas`, `numpy` – data handling  
  - `scikit-learn` – ML models & evaluation  
  - `matplotlib`, `seaborn` – visualization  
  - `nltk` – text preprocessing (stopwords, tokenization)   
  - `xgboost` – advanced ML algorithm  

---

## 📊 Methodology  
1. **Data Preprocessing**  
   - Lowercasing, punctuation removal, number & emoji removal  
   - Stopword removal  
   - TF-IDF & Bag-of-Words vectorization  

2. **Models Implemented**  
   - Naive Bayes (BoW & TF-IDF)  
   - Logistic Regression (TF-IDF)  
   - Support Vector Machine (SVM)  
   - XGBoost  

3. **Evaluation Metrics**  
   - Accuracy  
   - Confusion Matrix  
   - Classification Report (Precision, Recall, F1-score)  

4. **Additional Analysis**  
   - Emotion distribution plots   
   - Hyperparameter tuning with `GridSearchCV`  
   - Error analysis of misclassified samples 

---

## 🏆 Results  
| Model                 | Features | Accuracy |
|------------------------|----------|----------|
| Naive Bayes            | BoW      | 76.75%      |
| Naive Bayes            | TF-IDF   | 66.22%      |
| Logistic Regression    | TF-IDF   | 85.06%      |
| Linear SVM             | TF-IDF   | 88.75%      |
| XGBoost                | TF-IDF   | 87.91%      |

👉 Best model is SVM with accuracy around 88.75 percent

---

5. **Error Analysis**  
   - Created a **comparison dataframe** of true vs. predicted labels.  
   - Mapped text labels to numerical values for easier analysis.  
   - Extracted **misclassified samples** to understand common failure cases (e.g., confusing *joy* with *surprise*).  
     ```

   **Insights from Misclassifications**  
   - Some sentences contain **overlapping emotional tones**, making them difficult for ML models.  
   - Certain minority classes (e.g., *fear*, *disgust*) may have fewer training samples → leading to lower precision.  
   - Error analysis provides **guidance for future improvements** (e.g., balancing dataset, using advanced models like BERT).  

✅ This process ensures the project not only reports the best accuracy but also provides **transparency on where the model struggles**, making the work more **research-driven and credible**.  


