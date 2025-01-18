import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
print('running');

def load_and_preprocess_data(csv_file, max_words=10000, max_len=200):
    """
    Load and preprocess the review data from CSV file
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Convert sentiment to binary (assuming 'positive' is 1 and 'negative' is 0)
    df['sentiment'] = (df['column'] == 'positive').astype(int)
    
    # Initialize and fit the tokenizer
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(df['text'])
    
    # Convert text to sequences
    sequences = tokenizer.texts_to_sequences(df['text'])
    
    # Pad sequences to ensure uniform length
    X = pad_sequences(sequences, maxlen=max_len)
    y = df['sentiment'].values
    
    return X, y, tokenizer

def split_data(X, y, test_size=0.2, val_size=0.2):
    """
    Split data into train, validation, and test sets
    """
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Second split: separate validation set from remaining data
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_model(max_words, max_len, embedding_dim=100):
    """
    Create and compile the neural network model
    """
    model = Sequential([
        Embedding(max_words, embedding_dim, input_length=max_len),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    return model

def train_evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Train the model and evaluate its performance
    """
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping]
    )
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    
    return history, test_loss, test_accuracy

def main():
    # Configuration
    MAX_WORDS = 10000
    MAX_LEN = 200
    EMBEDDING_DIM = 100
    
    # Load and preprocess data
    X, y, tokenizer = load_and_preprocess_data('./train_data/reviews.csv', MAX_WORDS, MAX_LEN)
    
    # Split the data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # Create and train the model
    model = create_model(MAX_WORDS, MAX_LEN, EMBEDDING_DIM)
    history, test_loss, test_accuracy = train_evaluate_model(
        model, X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    return model, tokenizer, history

if __name__ == "__main__":
    model, tokenizer, history = main()