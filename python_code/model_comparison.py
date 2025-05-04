import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout, Flatten
from tensorflow.keras.regularizers import l2
import numpy as np

def create_model_basic(max_words, max_len, embedding_dim=100):
    """
    Create a basic model with no regularization
    """
    model = Sequential([
        Embedding(input_dim=max_words, output_dim=embedding_dim),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_model_dropout(max_words, max_len, embedding_dim=100):
    """
    Create a model with dropout regularization
    """
    model = Sequential([
        Embedding(input_dim=max_words, output_dim=embedding_dim),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_model_l2(max_words, max_len, embedding_dim=100):
    """
    Create a model with L2 regularization
    """
    l2_reg = l2(0.01)
    
    model = Sequential([
        Embedding(input_dim=max_words, output_dim=embedding_dim),
        Flatten(),
        Dense(64, activation='relu', kernel_regularizer=l2_reg),
        Dense(32, activation='relu', kernel_regularizer=l2_reg),
        Dense(1, activation='sigmoid', kernel_regularizer=l2_reg)
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def compare_models(X_train, y_train, X_val, y_val, X_test, y_test, max_words, max_len, embedding_dim=100):
    """
    Compare different model architectures and visualize their performance
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        max_words: Maximum number of words in vocabulary
        max_len: Maximum sequence length
        embedding_dim: Dimension of word embeddings
    
    Returns:
        Dictionary containing metrics for each model
    """
    models = {
        'basic': create_model_basic,
        'dropout': create_model_dropout,
        'l2': create_model_l2
    }
    
    results = {}
    
    for name, model_func in models.items():
        print(f"\nTraining {name} model...")
        model = model_func(max_words, max_len, embedding_dim)
        
        history = model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=32,
            validation_data=(X_val, y_val),
            verbose=1
        )
        
        # Evaluate on test set
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        
        # Store results
        results[name] = {
            'history': history.history,
            'test_loss': test_loss,
            'test_acc': test_acc
        }
    
    # Plot training and validation accuracy
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    for name, result in results.items():
        plt.plot(result['history']['accuracy'], label=f'{name} train')
        plt.plot(result['history']['val_accuracy'], label=f'{name} val')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for name, result in results.items():
        plt.plot(result['history']['loss'], label=f'{name} train')
        plt.plot(result['history']['val_loss'], label=f'{name} val')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print final test results
    print("\nFinal Test Results:")
    for name, result in results.items():
        print(f"{name} model - Test accuracy: {result['test_acc']:.4f}, Test loss: {result['test_loss']:.4f}")
    
    return results

def compare_l2_rates(X_train, y_train, X_val, y_val, X_test, y_test, max_words, max_len, embedding_dim=100, rates=None, epochs=10):
    """
    Compare different L2 regularization rates and plot their performance.
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        max_words: Maximum number of words in vocabulary
        max_len: Maximum sequence length
        embedding_dim: Dimension of word embeddings
        rates: List or array of L2 regularization rates to test (max 45)
        epochs: Number of epochs to train each model
    """
    if rates is None:
        rates = np.logspace(-5, 0, 20)  # Default: 20 rates from 1e-5 to 1
    rates = rates[:45]  # Limit to 45 rates
    val_accuracies = []
    test_accuracies = []
    for rate in rates:
        print(f"\nTraining L2 model with rate={rate:.5f}...")
        l2_reg = l2(rate)
        model = Sequential([
            Embedding(input_dim=max_words, output_dim=embedding_dim),
            Flatten(),
            Dense(64, activation='relu', kernel_regularizer=l2_reg),
            Dense(32, activation='relu', kernel_regularizer=l2_reg),
            Dense(1, activation='sigmoid', kernel_regularizer=l2_reg)
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_val, y_val),
            verbose=0
        )
        val_acc = history.history['val_accuracy'][-1]
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        val_accuracies.append(val_acc)
        test_accuracies.append(test_acc)
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(rates, val_accuracies, marker='o', label='Validation Accuracy')
    plt.plot(rates, test_accuracies, marker='x', label='Test Accuracy')
    plt.xscale('log')
    plt.xlabel('L2 Regularization Rate')
    plt.ylabel('Accuracy')
    plt.title('Effect of L2 Regularization Rate on Model Accuracy')
    plt.legend()
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.tight_layout()
    plt.show()
    return {'rates': rates, 'val_accuracies': val_accuracies, 'test_accuracies': test_accuracies} 