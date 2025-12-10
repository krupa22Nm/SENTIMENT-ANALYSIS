import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text):
    # Lowercase, remove URLs, mentions, hashtags, punctuation
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    
    # Remove stopwords & lemmatize
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(word) for word in text.split() 
              if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

def load_and_preprocess():
    # Example for tweets
    df = pd.read_csv('data/raw/tweets.csv', encoding='latin-1', 
                     names=['target', 'id', 'date', 'flag', 'user', 'text'])
    df['sentiment'] = df['target'].map({0: 'negative', 4: 'positive'})
    
    # Add neutral samples (e.g., from Amazon 3-star reviews)
    neutral_df = pd.read_csv('data/raw/amazon_neutral.csv')
    neutral_df['sentiment'] = 'neutral'
    
    df = pd.concat([df[['text', 'sentiment']], neutral_df])
    df['clean_text'] = df['text'].apply(clean_text)
    return df.dropna()