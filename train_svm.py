import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from utils import clean_text
import joblib

def run_svm_pipeline(data_path):
    # 1. Load Data
    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path, encoding='utf-8', on_bad_lines='skip')
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: Dataset not found at {data_path}. Please ensure the file exists.")
    
    # Validate required columns
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")

    # Drop rows with missing values in 'text' or 'label'
    df.dropna(subset=['text', 'label'], inplace=True)
    
    print("Cleaning text data...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], df['label'], test_size=0.2, random_state=42
    )
    
    # 3. Vectorization
    print("Vectorizing text with TF-IDF...")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # 4. Model Training
    print("Training SVM Model...")
    model = LinearSVC(C=1.0, dual=False) # Added dual=False for older sklearn versions or when n_samples > n_features
    model.fit(X_train_tfidf, y_train)
    
    # 5. Evaluation
    predictions = model.predict(X_test_tfidf)
    print("\nSVM Results:")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(classification_report(y_test, predictions))
    
    # Save models
    joblib.dump(model, 'svm_model.pkl')
    joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
    print("SVM model and TF-IDF vectorizer saved successfully.")

if __name__ == "__main__":
    # Path to your Kaggle dataset (e.g., 'train.csv')
    # Dataset link: https://www.kaggle.com/c/fake-news/data
    dataset_path = 'data/train.csv' 
    try:
        run_svm_pipeline(dataset_path)
    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)