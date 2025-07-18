import re
import emoji
import contractions
import pickle
import nltk
nltk.download('stopwords')

GAMING_SLANG = {
    "kys": "kill yourself", "stfu": "shut the fuck up", "gg": "good game"
}

def preprocess_gaming_chat(text):
    text = str(text).lower()
    text = contractions.fix(text)
    text = emoji.replace_emoji(text, replace='')
    words = [GAMING_SLANG.get(word.lower(), word) for word in text.split()]
    return ' '.join(words)

def load_models():
    with open('models/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('models/toxic_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return vectorizer, model