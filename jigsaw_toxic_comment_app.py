import streamlit as st
from utils import preprocess_gaming_chat, load_models

# Load models
vectorizer, model = load_models()

# App UI
st.title("⚠️ Toxic Comment Classifier")
comment = st.text_area("Enter a comment:")

if st.button("Analyze"):
    if comment:
        processed_text = preprocess_gaming_chat(comment)
        X = vectorizer.transform([processed_text])
        prediction = model.predict(X)
        st.success("✅ Non-Toxic" if prediction[0] == 0 else "🚨 Toxic")
    else:
        st.warning("Please enter a comment")
