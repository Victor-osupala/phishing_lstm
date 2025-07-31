import streamlit as st
import numpy as np
import pandas as pd
import json
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# App Title
st.set_page_config(page_title="Phishing URL Detector", layout="centered")
st.title("🔐 Phishing URL Detection using LSTM")
st.markdown("Enter a URL manually or upload a CSV file containing URLs to check for phishing threats.")

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    model = load_model("lstm_model/lstm_model.h5")
    with open("lstm_model/tokenizer.json") as f:
        tokenizer = tokenizer_from_json(json.load(f))
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# Preprocess input URL
def clean_url(url):
    url = re.sub(r'https?://', '', url)  # remove http/https
    return url.strip().lower()

def preprocess_urls(urls, tokenizer, max_len=150):
    urls_cleaned = [clean_url(url) for url in urls]
    sequences = tokenizer.texts_to_sequences(urls_cleaned)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post')
    return padded

# Prediction function
def predict(urls):
    processed = preprocess_urls(urls, tokenizer)
    preds = model.predict(processed)
    results = ['Phishing' if p > 0.5 else 'Legitimate' for p in preds]
    return preds.flatten(), results

# Sidebar Input
input_mode = st.sidebar.radio("Choose Input Method", ["Manual Input", "Upload CSV File"])

if input_mode == "Manual Input":
    user_url = st.text_input("🔗 Enter URL")
    if st.button("Detect"):
        if user_url:
            prob, label = predict([user_url])
            st.success(f"Prediction: **{label[0]}** ({prob[0]*100:.2f}% confidence)")
        else:
            st.warning("Please enter a URL to analyze.")

else:
    uploaded_file = st.file_uploader("📄 Upload CSV file with a column named `url`", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'url' not in df.columns:
                st.error("Uploaded CSV must contain a column named `url`.")
            else:
                if st.button("Run Detection on File"):
                    urls = df['url'].dropna().tolist()
                    probs, labels = predict(urls)
                    df['Prediction'] = labels
                    df['Confidence (%)'] = (probs * 100).round(2)
                    st.success("Detection complete!")
                    st.dataframe(df)
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Results CSV", data=csv, file_name="phishing_detection_results.csv", mime="text/csv")
        except Exception as e:
            st.error(f"An error occurred: {e}")
