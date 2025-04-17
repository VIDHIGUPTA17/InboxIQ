# # app.py
# from flask import Flask, request, jsonify
# import pickle
# import nltk
# # import string
# import re
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
# nltk.download('stopwords')
# nltk.download('punkt')

# app = Flask(__name__)


# ps = PorterStemmer()
# with open("model.pkl", "rb") as f:
#     model = pickle.load(f)

# with open("vectorizer.pkl", "rb") as f:
#     vectorizer = pickle.load(f)

# def preprocess(text):
#     # 1. Lowercase
#     text = text.lower()
    
#     # 2. Remove email, URLs, numbers, and special characters
#     text = re.sub(r'\S+@\S+', '', text)  # remove emails
#     text = re.sub(r'http\S+', '', text)  # remove URLs
#     text = re.sub(r'\d+', '', text)      # remove numbers
#     text = re.sub(r'[^a-z\s]', '', text) # remove special characters and punctuations
    
#     # 3. Tokenization (rds)
#     words = text.split()
    
#     # 4. Remove stopwords
#     words = [word for word in words if word not in stopwords.words('english')]
    
#     # 5. Stemming
#     words = [ps.stem(word) for word in words]
    
#     # 6. Join back into a single string
#     return " ".join(words)


# @app.route("/predict", methods=["POST"])
# def predict():
#     data = request.json
#     if not data or "text" not in data:
#         return jsonify({"error": "Missing 'text' in request"}), 400

#     text = data["text"]
#     clean = preprocess(text)
#     vector = vectorizer.transform([clean])
#     prediction = model.predict(vector)[0]
#     prob = model.predict_proba(vector)[0].max()

#     label = "Spam 🚨" if prediction == 1 else "Ham ✅"
#     return jsonify({
#         "prediction": label,
#         "confidence": round(prob, 2)
#     })

# if __name__ == "__main__":
#     app.run(debug=True)


# from flask import Flask, request, jsonify
# import pickle
# import numpy as np
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from feature_engineer import FeatureEngineer


# # Download NLTK resources if not already present
# nltk.download('stopwords')
# nltk.download('wordnet')

# # Load pickled files
# with open('spam_bundle_data.pkl', 'rb') as f:
#     bundle = pickle.load(f)

# model = bundle['model']
# scaler = bundle['scaler']
# fe = bundle['feature_engineer']


# # Init Flask app
# app = Flask(__name__)

# # Define preprocessor
# stop_words = set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()

# def enhanced_preprocess(text):
#     features = {
#         'num_emails': len(re.findall(r'\S+@\S+', text)),
#         'num_urls': len(re.findall(r'http\S+', text)),
#         'num_currency': len(re.findall(r'[$€£¥₹]', text)),
#         'num_exclamations': text.count('!'),
#         'num_uppercase': sum(1 for c in text if c.isupper()),
#         'num_digits': sum(1 for c in text if c.isdigit()),
#     }
    
#     text = text.lower()
#     text = re.sub(r'\S+@\S+', ' emailtoken ', text)
#     text = re.sub(r'http\S+', ' urltoken ', text)
#     text = re.sub(r'\d+', ' numbertoken ', text)
#     text = re.sub(r'[^a-z\s]', ' ', text)
    
#     words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words and len(w) > 2]
#     return ' '.join(words), features

# # Satisfaction meter & interpretation
# def get_satisfaction_meter(confidence):
#     if confidence >= 0.9:
#         return "🔴🔴🔴🔴🔴"
#     elif confidence >= 0.75:
#         return "🔴🔴🔴🔴⚫️"
#     elif confidence >= 0.6:
#         return "🟠🟠🟠⚫️⚫️"
#     elif confidence >= 0.4:
#         return "🟡🟡🟡⚫️⚫️"
#     elif confidence >= 0.2:
#         return "🟢🟢⚫️⚫️⚫️"
#     else:
#         return "🟢⚫️⚫️⚫️⚫️"

# def interpret_prediction(proba, threshold=0.4):
#     if proba >= 0.9:
#         return "Likely Spam 🚨"
#     elif proba >= 0.6:
#         return "Possibly Spam ⚠️"
#     elif proba >= threshold:
#         return "Uncertain ⚖️"
#     elif proba >= 0.2:
#         return "Likely Ham ✅"
#     else:
#         return "Confidently Ham ✅✅"

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     if 'text' not in data:
#         return jsonify({'error': 'Text field is required'}), 400
    
#     text = data['text']
#     cleaned_text, meta = enhanced_preprocess(text)
#     meta_array = np.array(list(meta.values())).reshape(1, -1)

#     features = fe.transform([cleaned_text], meta_array)
#     features_scaled = scaler.transform(features)

#     proba = model.predict_proba(features_scaled)[0]
#     prediction = interpret_prediction(proba[1])
#     meter = get_satisfaction_meter(proba[1])
#     # spam_conf = proba[1]
#     # ham_conf = proba[0]

#     result = {
#         'text': text,
#         'prediction': prediction,
#         'spam_confidence': f"{proba[1]:.2%}",
#         'ham_confidence': f"{proba[0]:.2%}",
#         'satisfaction_meter': meter
#     }
#     return jsonify(result)

# if __name__ == '__main__':
#     app.run(debug=True)




import streamlit as st
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import joblib
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# Setup
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

# Preprocessing setup
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def enhanced_preprocessor(text):
    text = text.lower()
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = [lemmatizer.lemmatize(word) for word in text.split()
             if word not in stop_words and len(word) > 2]
    return ' '.join(words)

# Feature Engineering
class FeatureEngineer:
    def __init__(self, max_features=3000, ngram_range=(1, 3)):
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True
        )
        self.svd = TruncatedSVD(n_components=300)

    def fit_transform(self, texts):
        tfidf_matrix = self.tfidf.fit_transform(texts)
        return self.svd.fit_transform(tfidf_matrix)

    def transform(self, texts):
        tfidf_matrix = self.tfidf.transform(texts)
        return self.svd.transform(tfidf_matrix)

# Confidence meter
def get_satisfaction_meter(confidence):
    if confidence >= 0.9:
        return "🔴🔴🔴🔴🔴"
    elif confidence >= 0.75:
        return "🔴🔴🔴🔴⚫️"
    elif confidence >= 0.6:
        return "🟠🟠🟠⚫️⚫️"
    elif confidence >= 0.4:
        return "🟡🟡🟡⚫️⚫️"
    elif confidence >= 0.2:
        return "🟢🟢⚫️⚫️⚫️"
    else:
        return "🟢⚫️⚫️⚫️⚫️"
    


def interpret_prediction(proba, threshold=0.4):
    if proba >= 0.9:
        return "Likely Spam 🚨"
    elif proba >= 0.6:
        return "Possibly Spam ⚠️"
    elif proba >= threshold:
        return "Uncertain ⚖️"
    elif proba >= 0.2:
        return "Likely Ham ✅"
    else:
        return "Confidently Ham ✅✅"

# Load pre-trained models (Replace these with your actual trained models)
# Assuming you have saved your models
fe = joblib.load("feature_engineer.pkl")
scaler = joblib.load("scaler.pkl")
ensemble = joblib.load("ensemble_model.pkl")

# Prediction Function
def predict_spam(text, threshold=0.4):
    cleaned_text = enhanced_preprocessor(text)
    features = fe.transform([cleaned_text])
    features_scaled = scaler.transform(features)
    proba = ensemble.predict_proba(features_scaled)[0]
    satisfaction_meter = get_satisfaction_meter(proba[1])
    final_prediction = interpret_prediction(proba[1], threshold)
    return {
        'text': text,
        'prediction': final_prediction,
        'spam_confidence': f"{proba[1]:.2%}",
        'ham_confidence': f"{proba[0]:.2%}",
        'satisfaction_meter': satisfaction_meter
    }

# ---------------- Streamlit UI ----------------
import streamlit as st
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Assuming all these are preloaded from your previous code:
# fe, scaler, enhanced_preprocessor, ensemble, get_satisfaction_meter, interpret_prediction

st.set_page_config(page_title="InboxIQ", page_icon="📧", layout="centered")

st.title("📨  InboxIQ")
st.markdown("A powerful ML-based tool to detect **Spam vs Ham** messages with confidence 🔍")

st.markdown("---")

# Text Input
user_input = st.text_area("✏️ Enter your message here:", height=150, placeholder="Type or paste your message...")

if st.button("🔍 Classify Message"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to classify.")
    else:
        result = predict_spam(user_input)

        # Output Section
        st.markdown("### 🧠 Prediction Result")
        st.markdown(f"**Prediction:** `{result['prediction']}`")
        st.markdown(f"**Spam Confidence:** `{result['spam_confidence']}`")
        st.markdown(f"**Ham Confidence:** `{result['ham_confidence']}`")

        # Visual Satisfaction Meter
        st.markdown("#### 📊 Satisfaction Meter")
        st.markdown(result['satisfaction_meter'])

        # Colored Box based on prediction
        if "Spam" in result['prediction']:
            st.error(f"🚨 **{result['prediction']}**", icon="🚨")
        elif "Ham" in result['prediction']:
            st.success(f"✅ **{result['prediction']}**", icon="✅")
        else:
            st.info(f"⚖️ **{result['prediction']}**", icon="⚖️")

st.markdown("---")
# st.markdown("🛡️ Built with `Scikit-learn`, `NLP`, `Streamlit`, and ❤️ by your AI assistant.")

