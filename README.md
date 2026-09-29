# ET-MLAM-04-Roman-Urdu-SMS-Spam-Classifier_CodeSaviours
# 📩 Roman Urdu SMS Spam Classifier using Logistic Regression

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A machine learning project that classifies SMS messages written in **Roman Urdu** (Urdu written in the Latin alphabet) as **Spam** or **Ham (legitimate)** using **Logistic Regression** and TF-IDF text features.

## 🔍 Overview
Most spam filters are built for English, but many users in Pakistan write in Roman Urdu, which has no standard spelling. This project builds a lightweight, interpretable classifier for that problem.

## 🎯 Problem Statement
Given a Roman Urdu SMS, predict whether it is **Ham (0)** or **Spam (1)**.

## 📊 Dataset
| Property | Details |
|---|---|
| Language | Roman Urdu |
| Task | Binary text classification |
| Total samples | `<add>` |
| Spam / Ham split | `<add>` |

## 🛠 Tech Stack
Python, pandas, numpy, scikit-learn, matplotlib, seaborn, Jupyter Notebook

## 🔄 Workflow
1. Load and explore data
2. Clean text (lowercase, remove URLs/punctuation/numbers)
3. TF-IDF vectorization
4. Train/test split
5. Train Logistic Regression
6. Evaluate (accuracy, precision, recall, F1, confusion matrix)
7. Predict on new messages

## ⚙ Installation
```bash
git clone https://github.com/<your-username>/roman-urdu-sms-spam-classifier.git
cd roman-urdu-sms-spam-classifier
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook
```

## 💻 Code

### Imports
```python
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

### Load Data
```python
df = pd.read_csv("roman_urdu_sms.csv")  # update filename/columns
print(df['label'].value_counts())
```

### Preprocessing
```python
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

df['clean_message'] = df['message'].apply(clean_text)
```

### Split and Vectorize
```python
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_message'], df['label'],
    test_size=0.2, random_state=42, stratify=df['label'])

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
```

### Train and Evaluate
```python
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
plt.show()
```

### Predict New Messages
```python
def predict_sms(message):
    vec = vectorizer.transform([clean_text(message)])
    return "Spam" if model.predict(vec)[0] == 1 else "Ham"

print(predict_sms("Mubarak ho! Aap ne 1 lakh ka inaam jeeta hai, abhi claim karein"))
print(predict_sms("Kal shaam ko milte hain, ghar aa jana"))
```

## 📈 Results
| Metric | Score |
|---|---|
| Accuracy | `<add>` |
| Precision | `<add>` |
| Recall | `<add>` |
| F1-Score | `<add>` |

## 📁 Project Structure
```
├── Project_4_Logistic_Regression_Roman_Urdu_SMS_Spam_Classifier.ipynb
├── data/roman_urdu_sms.csv
└── README.md
```

## 🚀 Future Improvements
- Compare with Naive Bayes, SVM, Random Forest
- Add a Roman Urdu spelling normalizer
- Handle class imbalance (class weights / SMOTE)
- Try LSTM or multilingual BERT
- Deploy with Streamlit or Flask

## 👤 Author
**Your Name** · [GitHub](https://github.com/your-username) · [LinkedIn](https://linkedin.com/in/your-profile)

## 📄 License
MIT License
