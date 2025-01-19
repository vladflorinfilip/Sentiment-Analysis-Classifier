import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import tensorflow as tf

def load_and_preprocess_data(csv_file, max_words=10000, max_len=200):
    """
    Load and preprocess the review data from CSV file with class balance verification
    
    Parameters:
    -----------
    csv_file : str
        Path to CSV file containing reviews and sentiments
    max_words : int, optional (default=10000)
        Maximum number of words to keep in vocabulary
    max_len : int, optional (default=200)
        Maximum length of each sequence
        
    Returns:
    --------
    X : array-like
        Padded sequences of reviews
    y : array-like
        Binary sentiment labels
    tokenizer : Tokenizer
        Fitted tokenizer object
    """
    # Read the CSV file
    print(f"\nLoading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    total_samples = len(df)
    print(f"Total number of reviews: {total_samples:,}")
    
    # Check class distribution before conversion
    sentiment_counts = df['sentiment'].value_counts()
    print("\nOriginal class distribution:")
    for sentiment, count in sentiment_counts.items():
        percentage = (count/total_samples) * 100
        print(f"{sentiment}: {count:,} reviews ({percentage:.1f}%)")
    
    # Verify expected class balance
    n_positive = (df['sentiment'] == 'Positive').sum()
    n_negative = (df['sentiment'] == 'Negative').sum()
    if n_positive != 5000 or n_negative != 5000:
        print("\nWARNING: Unexpected class distribution!")
        print(f"Expected: 5,000 positive and 5,000 negative")
        print(f"Found: {n_positive:,} positive and {n_negative:,} negative")
    
    # Convert sentiment to binary
    df['sentiment'] = (df['sentiment'] == 'Positive').astype(int)
    
    # Get text length statistics
    df['review_length'] = df['review'].str.len()
    print("\nReview length statistics:")
    print(f"Mean length: {df['review_length'].mean():.1f} characters")
    print(f"Median length: {df['review_length'].median():.1f} characters")
    print(f"Max length: {df['review_length'].max():,} characters")
    print(f"Min length: {df['review_length'].min():,} characters")
    
    # Initialize and fit tokenizer
    print("\nTokenizing reviews...")
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(df['review'])
    
    # Get vocabulary statistics
    vocab_size = len(tokenizer.word_index)
    print(f"Total unique words: {vocab_size:,}")
    print(f"Keeping top {max_words:,} words")
    
    # Convert text to sequences
    sequences = tokenizer.texts_to_sequences(df['review'])
    
    # Get sequence length statistics before padding
    seq_lengths = [len(seq) for seq in sequences]
    print("\nSequence length statistics (before padding):")
    print(f"Mean length: {np.mean(seq_lengths):.1f} words")
    print(f"Median length: {np.median(seq_lengths):.1f} words")
    print(f"Max length: {max(seq_lengths):,} words")
    print(f"Min length: {min(seq_lengths):,} words")
    
    # Calculate how many sequences will be truncated
    n_truncated = sum(len(seq) > max_len for seq in sequences)
    if n_truncated > 0:
        print(f"\nWARNING: {n_truncated:,} reviews ({(n_truncated/total_samples)*100:.1f}%) "
              f"will be truncated to {max_len} words")
    
    # Pad sequences
    print(f"\nPadding sequences to length {max_len}...")
    X = pad_sequences(sequences, maxlen=max_len)
    y = df['sentiment'].values
    
    # Final verification of processed data
    print("\nFinal processed data shape:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Class balance in processed data: {np.mean(y)*100:.1f}% positive")
    
    return X, y, tokenizer

def split_data(X, y, test_size=0.2, val_size=0.2):
    """
    Split data into train, validation, and test sets
    """
    # Shuffle the data to ensure randomness in splits
    X, y = shuffle(X, y, random_state=42)

    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Second split: separate validation set from remaining data
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size_adjusted, random_state=42)


    # Check label distribution in each set
    print("Label distribution:")
    print(f"Train set: {np.unique(y_train, return_counts=True)}")
    print(f"Validation set: {np.unique(y_val, return_counts=True)}")
    print(f"Test set: {np.unique(y_test, return_counts=True)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_model(max_words, max_len, embedding_dim=100):
    """
    Create and compile the neural network model
    """
    # Create Adam optimizer with recommended parameters
    adam_optimizer = tf.keras.optimizers.Adam(
        learning_rate=5e-5,  # Default learning rate
        beta_1=0.9,     # Exponential decay rate for 1st moment estimates
        beta_2=0.999,   # Exponential decay rate for 2nd moment estimates
        epsilon=1e-7,   # Small constant for numerical stability
        amsgrad=False,   # Whether to apply AMSGrad variant
        clipvalue=1.0   # Clips gradients to the range [-1.0, 1.0]
    )

    model = Sequential([
        Embedding(max_words, embedding_dim, input_length=max_len),
        LSTM(64, return_sequences=True, kernel_regularizer=l2(0.05)),
        LSTM(32, kernel_regularizer=l2(0.05)),
        Dense(64, activation='relu', kernel_regularizer=l2(0.05)),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=adam_optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test,):
    """
    Train the model and evaluate its performance
    """
    print("\nTraining final model...")

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    # Train the model
    class_weights = {0: 1., 1: 1.1}  # Slightly higher weight for positive class
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks= early_stopping,
        class_weight=class_weights
    )
    
    # Evaluate on test set
    final_metrics = evaluate_final_model(
        model, X_train, y_train, X_val, y_val, X_test, y_test
    )
    
    return history, final_metrics

def evaluate_final_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Evaluate model on all datasets and return detailed metrics
    """
    # Training metrics
    train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    train_error = 1 - train_accuracy
    
    # Validation metrics
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    val_error = 1 - val_accuracy
    
    # Test metrics
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    test_error = 1 - test_accuracy
    
    from sklearn.metrics import confusion_matrix

    # Get predictions for the validation set
    y_val_pred = model.predict(X_val)
    y_val_pred = (y_val_pred > 0.5).astype(int)

    # Generate confusion matrix
    cm = confusion_matrix(y_val, y_val_pred)
    print(f'Confusion Matrix:\n{cm}')

    print("\nFinal model metrics:")
    print(f"Training Error: {train_error:.4f}")
    print(f"Validation Error: {val_error:.4f}")
    print(f"Test Error: {test_error:.4f}")
    
    print(f"\nTraining Loss: {train_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    return {
        'train_error': train_error,
        'val_error': val_error,
        'test_error': test_error,
        'train_loss': train_loss,
        'val_loss': val_loss,
    }


def main():
    # Configuration
    MAX_WORDS = 10000
    MAX_LEN = 200
    EMBEDDING_DIM = 100
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y, tokenizer = load_and_preprocess_data('./train_data/reviews.csv', MAX_WORDS, MAX_LEN)
    
    # Split the data
    print("\nSplitting data into 60-20-20...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # Create and train the model
    model = create_model(MAX_WORDS, MAX_LEN, EMBEDDING_DIM)
    history, final_metrics = train_evaluate_model(
        model, X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    # Calculate differences to assess overfitting
    train_test_diff = final_metrics['test_error'] - final_metrics['train_error']
    print(f"\nDifference between test and training error: {train_test_diff:.4f}")
    
    return model, tokenizer, history

if __name__ == "__main__":
    model, tokenizer, history = main()