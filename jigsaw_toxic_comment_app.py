import streamlit as st
import pandas as pd
import numpy as np
from utils import preprocess_gaming_chat, load_models
import time

# Set page config
st.set_page_config(
    page_title="Toxic Comment Filter",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load models (cache to avoid reloading)
@st.cache_resource
def load_models_cached():
    return load_models()

vectorizer, model = load_models_cached()

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #F5F5F5;
    }
    .stTextArea textarea {
        height: 150px;
    }
    .toxic {
        color: red;
        font-weight: bold;
    }
    .non-toxic {
        color: green;
        font-weight: bold;
    }
    .header {
        color: #4B4B4B;
    }
    .sidebar .sidebar-content {
        background-color: #E1E1E1;
    }
    </style>
    """, unsafe_allow_html=True)

# App header
st.title("⚠️ Toxic Comment Classification")
st.markdown("""
This tool analyzes text comments and classifies them into different toxicity categories.
The model was trained on the Jigsaw Toxic Comment Classification dataset.
""")

# Sidebar
st.sidebar.header("About")
st.sidebar.info("""
This application uses machine learning to detect toxic comments in online conversations, 
with special focus on gaming chat terminology.

**Categories detected:**
- Toxic
- Severe Toxic
- Obscene
- Threat
- Insult
- Identity Hate
""")

st.sidebar.header("How to Use")
st.sidebar.info("""
1. Enter text in the input box
2. Click 'Analyze' button
3. View toxicity predictions
""")

# Main content
tab1, tab2 = st.tabs(["Single Comment Analysis", "Batch Analysis"])

with tab1:
    st.subheader("Analyze a Single Comment")
    comment = st.text_area("Enter your comment here:", 
                          placeholder="Type or paste your text here...")
    
    if st.button("Analyze", key="analyze_single"):
        if comment.strip() == "":
            st.warning("Please enter a comment to analyze.")
        else:
            with st.spinner("Analyzing comment..."):
                # Preprocess
                processed_text = preprocess_gaming_chat(comment)
                
                # Vectorize
                X = vectorizer.transform([processed_text])
                
                # Predict
                prediction = model.predict(X)
                probabilities = model.predict_proba(X)
                
                # Display results
                st.subheader("Results")
                
                # Overall toxicity
                toxic_prob = probabilities[0][1][1]  # Probability of toxic
                if toxic_prob > 0.5:
                    st.markdown(f"<p class='toxic'>Toxic (confidence: {toxic_prob:.2%})</p>", 
                               unsafe_allow_html=True)
                else:
                    st.markdown(f"<p class='non-toxic'>Non-Toxic (confidence: {1-toxic_prob:.2%})</p>", 
                               unsafe_allow_html=True)
                
                # Detailed categories
                categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
                results = pd.DataFrame({
                    'Category': categories,
                    'Prediction': ['Yes' if p[1] > 0.5 else 'No' for p in probabilities],
                    'Probability': [f"{p[1]:.2%}" for p in probabilities]
                })
                
                st.table(results)
                
                # Show processed text
                with st.expander("Show processed text"):
                    st.write(processed_text)

with tab2:
    st.subheader("Analyze Multiple Comments")
    uploaded_file = st.file_uploader("Upload a CSV file with comments:", 
                                   type=["csv"], 
                                   key="file_uploader")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if 'comment_text' not in df.columns:
                st.error("The CSV file must contain a 'comment_text' column.")
            else:
                st.success("File uploaded successfully!")
                st.write(f"Found {len(df)} comments.")
                
                if st.button("Analyze All", key="analyze_batch"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    results = []
                    for i, row in enumerate(df.itertuples()):
                        # Update progress
                        progress = (i + 1) / len(df)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing {i+1} of {len(df)} comments...")
                        
                        # Process and predict
                        processed_text = preprocess_gaming_chat(row.comment_text)
                        X = vectorizer.transform([processed_text])
                        probabilities = model.predict_proba(X)
                        
                        # Get probabilities for each category
                        result = {
                            'original_text': row.comment_text,
                            'processed_text': processed_text,
                            'is_toxic': any(p[1] > 0.5 for p in probabilities),
                            'toxic_prob': probabilities[0][1][1],
                            'severe_toxic_prob': probabilities[1][1][1],
                            'obscene_prob': probabilities[2][1][1],
                            'threat_prob': probabilities[3][1][1],
                            'insult_prob': probabilities[4][1][1],
                            'identity_hate_prob': probabilities[5][1][1]
                        }
                        results.append(result)
                    
                    # Create results DataFrame
                    results_df = pd.DataFrame(results)
                    
                    # Display summary
                    toxic_count = results_df['is_toxic'].sum()
                    st.subheader("Summary")
                    st.write(f"Found {toxic_count} toxic comments out of {len(df)} ({toxic_count/len(df):.1%})")
                    
                    # Show detailed results
                    st.subheader("Detailed Results")
                    st.dataframe(results_df)
                    
                    # Download button
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name='toxic_comment_results.csv',
                        mime='text/csv'
                    )
                    
                    progress_bar.empty()
                    status_text.empty()
                    st.success("Analysis complete!")
        
        except Exception as e:
            st.error(f"Error reading file: {e}")

# Footer
st.markdown("---")
st.markdown("""
**Note:** This is a machine learning model and may not be 100% accurate. 
Use it as a tool to assist with moderation, not as a definitive judgment.
""")