import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
import numpy as np

# ---------------------------
# Download NLTK stopwords (only first time)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# ---------------------------
# Load the trained model and vectorizer
try:
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
except Exception as e:
    st.error(f"❌ Error loading model/vectorizer: {e}")
    st.stop()

# ---------------------------
# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)         # remove punctuation/numbers
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# ---------------------------
# Streamlit UI
st.set_page_config(page_title="CV Classifier", layout="centered")
st.title("📄 CV Job Category Classifier")
st.markdown("Predict the job category of a given CV using a trained ML model.")

user_input = st.text_area("📝 Paste the CV content here:", height=300)

if st.button("🔍 Predict Category"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some CV content.")
    else:
        cleaned = clean_text(user_input)
        transformed = vectorizer.transform([cleaned])

        # Predict
        try:
            prediction = model.predict(transformed)[0]
            st.success(f"🎯 Predicted Job Category: **{prediction}**")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
