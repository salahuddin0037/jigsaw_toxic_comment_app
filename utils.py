import re
import emoji
import contractions
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import pickle
import string

# Download NLTK data
nltk.download('stopwords')

# Slang dictionary for gaming terms
GAMING_SLANG = {
    "kys": "kill yourself", "stfu": "shut the fuck up", "gg": "good game",
    "glhf": "good luck have fun", "noob": "new player", "rekt": "wrecked",
    "pwned": "dominated", "l2p": "learn to play", "wp": "well played",
    "afk": "away from keyboard", "brb": "be right back", "ggwp": "good game well played",
    "gl": "good luck", "hf": "have fun", "imo": "in my opinion",
    "ns": "nice shot", "nt": "nice try", "omw": "on my way"
}

def preprocess_gaming_chat(text):
    """Enhanced preprocessing for gaming chat with slang normalization"""
    if not isinstance(text, str):
        return ""
    
    text = str(text).lower()
    text = contractions.fix(text)
    text = emoji.replace_emoji(text, replace='')
    
    # Replace gaming slang
    words = text.split()
    words = [GAMING_SLANG.get(word.lower(), word) for word in words]
    text = ' '.join(words)
    
    # Remove repeated characters (e.g., "noooooob" -> "noob")
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    
    # Remove excessive punctuation (e.g., "!!!!" -> "!")
    text = re.sub(r'[!?]{2,}', '', text)
    
    # Remove non-alphanumeric characters (keeping basic punctuation)
    text = re.sub(r'[^a-zA-Z0-9\\s.,!?]', '', text)
    
    return text.strip()

def load_models():
    """Load pre-trained model and vectorizer"""
    try:
        with open('models/vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('models/toxic_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return vectorizer, model
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None