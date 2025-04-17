
# 📬 InboxIQ - Spam Message Classifier

InboxIQ is a powerful ML-powered web application built with **Streamlit** that detects whether a message is **Spam** or **Ham (Not Spam)**. It uses Natural Language Processing (NLP) and ensemble learning techniques to provide accurate and interpretable results along with confidence indicators and visualizations.

---

## ✨ Features

- 🔍 Classify text messages as Spam or Ham
- 📊 Satisfaction meter based on prediction confidence
- 🧠 Interpretable prediction levels (Likely, Possibly, Uncertain, etc.)
- ☁️ WordCloud for visualizing spam-indicative words
- 📈 Uses TF-IDF, Truncated SVD and Ensemble Modeling (SVM + Naive Bayes + Logistic Regression)

---

## 🛠️ Full Setup Instructions

Follow the steps below to set up the project and run the application locally.

---

### ✅ Prerequisites

Ensure that you have the following installed on your system:

- Python 3.8 or later
- pip (Python package installer)
- git (optional, for cloning the repo)
- virtualenv (recommended for managing dependencies)

---

### 📥 1. Clone the Repository

```bash
git clone https://github.com/your-username/InboxIQ.git
cd InboxIQ
```

Or download the ZIP and extract manually.

---

### 🧪 2. Set Up a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate it:

- **Windows**:
  ```bash
  venv\Scripts\activate
  ```
- **Mac/Linux**:
  ```bash
  source venv/bin/activate
  ```

---

### 📦 3. Install Required Dependencies

```bash
pip install -r requirements.txt
```

Dependencies include:
- streamlit
- scikit-learn
- joblib
- nltk
- matplotlib
- wordcloud
- numpy
- pandas

---

### 📚 4. Download NLTK Data

Run this in Python shell or script:

```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
```

---

### 🧠 5. Add Pre-trained Models

Ensure the following files are in the project directory:
- `feature_engineer.pkl`
- `scaler.pkl`
- `ensemble_model.pkl`

---

### ▶️ 6. Run the Application

```bash
streamlit run app.py
```

Visit `http://localhost:8501` if it doesn’t open automatically.

---

### 🧹 7. Deactivate the Virtual Environment (Optional)

```bash
deactivate
```

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgements

- Streamlit Team for the UI framework
- NLTK for natural language tools
- Scikit-learn for machine learning pipelines

---

