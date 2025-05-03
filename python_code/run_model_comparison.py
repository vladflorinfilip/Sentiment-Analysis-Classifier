import os
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
import random
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model_comparison import compare_models

def clean_text(text):
    text = text.lower() 
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\b\w\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_texts(texts, max_words=10000, max_len=100):
    texts = [clean_text(text) for text in texts]
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    
    vocab_size = len(tokenizer.word_index)
    print(f"Total unique words: {vocab_size:,}")
    print(f"Keeping top {max_words:,} words")
    
    sequences = tokenizer.texts_to_sequences(texts)
    
    seq_lengths = [len(seq) for seq in sequences]
    print("\nSequence length statistics (before padding):")
    print(f"Mean length: {np.mean(seq_lengths):.1f} words")
    print(f"Median length: {np.median(seq_lengths):.1f} words")
    print(f"Max length: {max(seq_lengths):,} words")
    print(f"Min length: {min(seq_lengths):,} words")
    
    n_truncated = sum(len(seq) > max_len for seq in sequences)
    total_samples = len(sequences)
    if n_truncated > 0:
        print(f"\nWARNING: {n_truncated:,} reviews ({(n_truncated/total_samples)*100:.1f}%) "
              f"will be truncated to {max_len} words")
    
    X = pad_sequences(sequences, maxlen=max_len)
    return X, tokenizer, sequences

def train_word2vec(texts, vector_size=100, window=5, min_count=2, workers=4):
    """
    Trains word2vec on tokenized text
    """
    tokenized_texts = [text.split() for text in texts]
    model = Word2Vec(tokenized_texts, vector_size=vector_size, window=window, min_count=min_count, workers=workers, sg=1)
    model.save("word2vec_model.model")
    
    return model

def split_data(X, y, test_size=0.2, val_size=0.2):
    """
    Split data into train, validation, and test sets
    """
    X, y = shuffle(X, y, random_state=42)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=42)
    print("Label distribution:")
    print(f"Train set: {np.unique(y_train, return_counts=True)}")
    print(f"Validation set: {np.unique(y_val, return_counts=True)}")
    print(f"Test set: {np.unique(y_test, return_counts=True)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def main():
    # Configuration
    DATA_FILE = 'data/reviews.csv'  # Update this path to your actual data file
    MAX_WORDS = 10000
    MAX_LEN = 200
    EMBEDDING_DIM = 100
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(DATA_FILE)
    texts = df['review'].values
    y = (df['sentiment'] == 'Positive').astype(int).values
    
    # Tokenize texts
    print("\nTokenizing texts...")
    X, tokenizer, sequences = tokenize_texts(texts, max_words=MAX_WORDS, max_len=MAX_LEN)
    
    # Train Word2Vec model
    print("\nTraining Word2Vec model...")
    w2v_model = train_word2vec(texts, vector_size=EMBEDDING_DIM)
    
    # Create embedding matrix
    embedding_matrix = np.zeros((MAX_WORDS, EMBEDDING_DIM))
    for word, i in tokenizer.word_index.items():
        if i < MAX_WORDS:
            if word in w2v_model.wv:
                embedding_matrix[i] = w2v_model.wv[word]
    
    # Split data
    print("\nSplitting data into train, validation, and test sets...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # Run model comparison
    print("\nRunning model comparison...")
    results = compare_models(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        max_words=MAX_WORDS,
        max_len=MAX_LEN,
        embedding_dim=EMBEDDING_DIM
    )
    
    # Save results
    print("\nSaving results...")
    results_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Test Accuracy': [r['test_acc'] for r in results.values()],
        'Test Loss': [r['test_loss'] for r in results.values()]
    })
    results_df.to_csv('model_comparison_results.csv', index=False)
    print("Results saved to model_comparison_results.csv")

if __name__ == "__main__":
    main() 