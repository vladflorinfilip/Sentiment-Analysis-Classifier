import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from neural_network import load_and_preprocess_data

def load_and_preprocess_word2vec(csv_file, max_words=10000, max_len=200):
    """
    Load and preprocess data for Word2Vec embeddings
    """
    # Load and preprocess data
    X, y, tokenizer = load_and_preprocess_data(csv_file, max_words, max_len)
    
    # Train Word2Vec model
    texts = [text.split() for text in pd.read_csv(csv_file)['review']]
    w2v_model = Word2Vec(texts, vector_size=100, window=5, min_count=2, workers=4, sg=1)
    
    # Create embedding matrix
    embedding_matrix = np.zeros((max_words, 100))
    for word, i in tokenizer.word_index.items():
        if i < max_words:
            if word in w2v_model.wv:
                embedding_matrix[i] = w2v_model.wv[word]
    
    return X, y, embedding_matrix, tokenizer

def load_and_preprocess_glove(csv_file, glove_path, max_words=10000, max_len=200):
    """
    Load and preprocess data for GloVe embeddings
    """
    # Load and preprocess data
    X, y, tokenizer = load_and_preprocess_data(csv_file, max_words, max_len)
    
    # Load GloVe embeddings
    embeddings_index = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    
    # Create embedding matrix
    embedding_matrix = np.zeros((max_words, 100))
    for word, i in tokenizer.word_index.items():
        if i < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    
    return X, y, embedding_matrix, tokenizer 