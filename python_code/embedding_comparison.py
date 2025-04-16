import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from embedding_utils import load_and_preprocess_word2vec, load_and_preprocess_glove
from neural_network import split_data

def compare_embeddings(csv_file, glove_path):
    """
    Compare Word2Vec and GloVe embeddings performance.
    """
    # Process with both embeddings
    X_w2v, y_w2v, w2v_matrix, _ = load_and_preprocess_word2vec(csv_file)
    X_glove, y_glove, glove_matrix, _ = load_and_preprocess_glove(csv_file, glove_path)
    
    # Split data for both embeddings
    X_w2v_train, X_w2v_val, X_w2v_test, y_w2v_train, y_w2v_val, y_w2v_test = split_data(X_w2v, y_w2v)
    X_glove_train, X_glove_val, X_glove_test, y_glove_train, y_glove_val, y_glove_test = split_data(X_glove, y_glove)
    
    # Simple LSTM model template
    def create_model(embedding_matrix):
        model = Sequential([
            Embedding(10000, 100, weights=[embedding_matrix], trainable=False),
            LSTM(64),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    # Train and evaluate both models
    print("\nTraining Word2Vec model:")
    w2v_model = create_model(w2v_matrix)
    w2v_history = w2v_model.fit(
        X_w2v_train, y_w2v_train,
        epochs=3,
        batch_size=32,
        validation_data=(X_w2v_val, y_w2v_val)
    )
    
    print("\nTraining GloVe model:")
    glove_model = create_model(glove_matrix)
    glove_history = glove_model.fit(
        X_glove_train, y_glove_train,
        epochs=3,
        batch_size=32,
        validation_data=(X_glove_val, y_glove_val)
    )
    
    # Compare results
    results = {
        'word2vec': {
            'val_accuracy': w2v_history.history['val_accuracy'][-1],
            'train_accuracy': w2v_history.history['accuracy'][-1]
        },
        'glove': {
            'val_accuracy': glove_history.history['val_accuracy'][-1],
            'train_accuracy': glove_history.history['accuracy'][-1]
        }
    }
    
    print("\nFinal Results:")
    print(f"Word2Vec - Train: {results['word2vec']['train_accuracy']:.4f}, "
          f"Val: {results['word2vec']['val_accuracy']:.4f}")
    print(f"GloVe - Train: {results['glove']['train_accuracy']:.4f}, "
          f"Val: {results['glove']['val_accuracy']:.4f}")
    
    return results

if __name__ == "__main__":
    CSV_FILE = "../train_data/reviews.csv"
    GLOVE_PATH = "../glove.6B/glove.6B.50d.txt"  # Make sure this path is correct
    
    # Run comparison
    results = compare_embeddings(CSV_FILE, GLOVE_PATH) 