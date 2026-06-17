import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Pre-compile regex patterns for better performance
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
HTML_PATTERN = re.compile(r'<.*?>')
PUNCT_PATTERN = re.compile(f'[{re.escape(string.punctuation)}]')
NUM_PATTERN = re.compile(r'\d+')

def clean_text(text: str) -> str:
    """
    Cleans the input text by removing HTML tags, punctuation, numbers, 
    and applying lemmatization.
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = URL_PATTERN.sub('', text)
    
    # Remove HTML tags
    text = HTML_PATTERN.sub('', text)
    
    # Remove punctuation and numbers
    text = PUNCT_PATTERN.sub('', text)
    text = NUM_PATTERN.sub('', text)
    
    # Tokenization and Lemmatization
    tokens = nltk.word_tokenize(text)
    cleaned_tokens = [
        lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
    ]
    
    return " ".join(cleaned_tokens)