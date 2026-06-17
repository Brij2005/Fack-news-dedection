import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import clean_text
import os

# Cache to avoid reloading heavy models on every prediction
_MODEL_CACHE = {}

def predict_fake_news(text, model_type='svm'):
    """
    Predicts if the given news text is Fake or Real.
    model_type: 'svm' or 'lstm'
    """
    cleaned = clean_text(text)
    
    if model_type == 'svm':
        if 'svm' not in _MODEL_CACHE:
            if not os.path.exists('svm_model.pkl') or not os.path.exists('tfidf_vectorizer.pkl'):
                print("Error: SVM model files not found. Please run train_svm.py first.")
                return
            _MODEL_CACHE['svm'] = joblib.load('svm_model.pkl')
            _MODEL_CACHE['tfidf'] = joblib.load('tfidf_vectorizer.pkl')
        
        vectorized_text = _MODEL_CACHE['tfidf'].transform([cleaned])
        prediction = _MODEL_CACHE['svm'].predict(vectorized_text)[0]
        confidence = "N/A (LinearSVC)"
        
    elif model_type == 'lstm':
        if 'lstm' not in _MODEL_CACHE:
            if not os.path.exists('fake_news_lstm.h5') or not os.path.exists('lstm_tokenizer.pkl'):
                print("Error: LSTM model files not found. Please run train_lstm.py first.")
                return
            _MODEL_CACHE['lstm'] = tf.keras.models.load_model('fake_news_lstm.h5')
            _MODEL_CACHE['tokenizer'] = joblib.load('lstm_tokenizer.pkl')
        
        # Preprocessing for LSTM
        sequence = _MODEL_CACHE['tokenizer'].texts_to_sequences([cleaned])
        padded = pad_sequences(sequence, maxlen=300)
        
        prob = _MODEL_CACHE['lstm'].predict(padded, verbose=0)[0][0]
        prediction = 1 if prob > 0.5 else 0
        confidence = f"{prob*100:.2f}%" if prediction == 1 else f"{(1-prob)*100:.2f}%"
    else:
        print("Invalid model type selection.")
        return

    result = "FAKE" if prediction == 1 else "REAL"
    print(f"\n[{model_type.upper()}] Result: {result} (Confidence: {confidence})")
    print(f"Snippet: {text[:75]}...")

if __name__ == "__main__":
    print("Fake News Detection System Loaded.")
    while True:
        user_input = input("\nEnter news text (or type 'exit' to stop): ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        print("\nChoose Model:")
        print("1. SVM (Fast)")
        print("2. LSTM (Deep Learning)")
        choice = input("Enter choice (1/2): ")
        
        if choice == '1':
            predict_fake_news(user_input, model_type='svm')
        elif choice == '2':
            predict_fake_news(user_input, model_type='lstm')
        else:
            print("Invalid choice.")