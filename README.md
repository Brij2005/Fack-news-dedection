# Fake News Detection using NLP

This project implements two machine learning models for detecting fake news: a Bidirectional LSTM (Long Short-Term Memory) neural network and a Support Vector Machine (SVM) with TF-IDF features. Both models utilize Natural Language Processing (NLP) techniques for text cleaning and feature extraction.

## Goal

The primary goal is to classify news articles as either "fake" or "real" based on their textual content.

## Features

- **Text Preprocessing:** Includes lowercasing, URL removal, HTML tag removal, punctuation and number removal, stop word removal, and lemmatization.
- **TF-IDF Vectorization:** Used for the SVM model to convert text into numerical feature vectors.
- **Word Embeddings:** Used for the LSTM model to represent words in a dense vector space.
- **Bidirectional LSTM:** A deep learning model capable of capturing long-range dependencies in text.
- **Linear SVM:** A robust traditional machine learning model for text classification.
- **Model Persistence:** Trained models and vectorizers/tokenizers are saved for future use.

## Setup Instructions

Follow these steps to set up and run the project:

### 1. Clone the Repository (if applicable)
If your project is in a repository, clone it. Otherwise, ensure all files are in the specified directory.

### 2. Create a Virtual Environment (Recommended)
It's good practice to use a virtual environment to manage dependencies.

```bash
python -m venv venv
source venv/bin/activate  # On Linux/macOS
venv\Scripts\activate     # On Windows
```

### 3. Install Required Libraries
Install all necessary Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 4. Download NLTK Data
The `utils.py` script automatically downloads the required NLTK data (`stopwords`, `wordnet`, `punkt`) the first time it's run. Ensure you have an internet connection during the first execution.

### 5. Obtain the Dataset
This project expects a CSV dataset named `train.csv` to be located in a `data/` subdirectory within your project root.

You can download the "Fake News" dataset from Kaggle:
https://www.kaggle.com/c/fake-news/data

1. Download `train.csv` from the Kaggle competition page.
2. Create a folder named `data` in the same directory as your Python scripts.
3. Place the downloaded `train.csv` file into the `data/` folder.

The expected path for the dataset is `data/train.csv`.

## How to Run

You can run each model independently.

### Run the LSTM Model

To train and evaluate the Bidirectional LSTM model:

```bash
python c:\Users\Krishna Sharma\Desktop\Fack news dedection\train_lstm.py
```

This will train the LSTM model, print its summary and evaluation metrics, and save the trained model (`fake_news_lstm.h5`) and its tokenizer (`lstm_tokenizer.pkl`) in the project root directory.

### Run the SVM Model

To train and evaluate the Linear SVM model:

```bash
python c:\Users\Krishna Sharma\Desktop\Fack news dedection\train_svm.py
```

This will train the SVM model, print its accuracy and classification report, and save the trained model (`svm_model.pkl`) and the TF-IDF vectorizer (`tfidf_vectorizer.pkl`) in the project root directory.

## Output

Upon successful execution, you will find the following files generated in your project directory:

- `fake_news_lstm.h5`: The trained Keras LSTM model.
- `lstm_tokenizer.pkl`: The fitted Keras Tokenizer object for the LSTM model.
- `svm_model.pkl`: The trained Scikit-learn LinearSVC model.
- `tfidf_vectorizer.pkl`: The fitted Scikit-learn TfidfVectorizer.