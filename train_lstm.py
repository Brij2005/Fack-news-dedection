import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utils import clean_text
import joblib # Import joblib to save tokenizer

def run_lstm_pipeline(data_path):
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    MAX_VOCAB = 20000
    MAX_SEQUENCE_LENGTH = 300
    EMBEDDING_DIM = 100

    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path, encoding='utf-8', on_bad_lines='skip')
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: Dataset not found at {data_path}. Please ensure the file exists.")
    
    # Validate required columns
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")

    # Drop rows with missing values in 'text' or 'label' after validation
    df.dropna(subset=['text', 'label'], inplace=True)
    
    df['cleaned_text'] = df['text'].apply(clean_text)

    # 2. Tokenization
    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['cleaned_text'])
    sequences = tokenizer.texts_to_sequences(df['cleaned_text'])
    
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # 3. Model Architecture
    model = Sequential([
        Embedding(MAX_VOCAB + 1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.5),
        Bidirectional(LSTM(32)),
        Dense(24, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    # 4. Training
    print("Starting LSTM training...")
    # Added EarlyStopping to prevent overfitting
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=2, 
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train, 
        epochs=10, 
        batch_size=64, 
        validation_data=(X_test, y_test),
        callbacks=[early_stop]
    )

    # 5. Evaluation
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nLSTM Test Accuracy: {accuracy:.4f}")

    # Generate detailed metrics
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print("\nDetailed LSTM Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save models and tokenizer
    model.save('fake_news_lstm.h5')
    joblib.dump(tokenizer, 'lstm_tokenizer.pkl') # Save the tokenizer
    print("LSTM model and tokenizer saved successfully.")

if __name__ == "__main__":
    dataset_path = 'data/train.csv'
    try:
        run_lstm_pipeline(dataset_path)
    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)