# Sentiment Analysis Classifier

This project implements a comprehensive sentiment analysis system using multiple approaches: a Naive Bayes classifier and a Deep Learning model, with support for different word embedding techniques (Word2Vec and GloVe). The system analyzes text reviews and classifies them as either positive or negative sentiments, providing a robust comparison between different machine learning approaches.

## Project Structure

```
Sentiment_Analysis_Classifier/
├── README.md
├── requirements.txt
├── notebook/
│   ├── data_analysis.ipynb        # Jupyter notebook for data analysis and visualization
│   └── word2vec_model.model      # Pre-trained Word2Vec model
├── python_code/
│   ├── neural_network.py         # LSTM-based sentiment classifier
│   ├── naive_bayes.py           # Naive Bayes sentiment classifier
│   ├── embedding_utils.py       # Utility functions for word embeddings
│   ├── embedding_comparison.py  # Comparison of different embedding methods
│   ├── nn_test.py              # Neural network testing utilities
│   ├── comparison.py           # Model comparison utilities
│   ├── tokenizer.json          # Saved tokenizer for text processing
│   └── sentiment_model.keras   # Pre-trained sentiment model
├── data/
│   └── reviews.csv             # Training and testing data
├── models/
│   ├── saved_nb_model/         # Saved Naive Bayes models
│   └── saved_nn_model/         # Saved Neural Network models
└── glove.6B/                   # GloVe word embeddings
```

## Requirements

- Python 3.12
- scikit-learn
- TensorFlow
- pandas
- numpy
- matplotlib
- seaborn
- jupyter
- gensim (for Word2Vec)

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Data Analysis and Visualization

The project includes a comprehensive Jupyter notebook (`notebook/data_analysis.ipynb`) that provides:
- Exploratory data analysis
- Data preprocessing visualization
- Model performance comparison
- Word embedding visualizations
- Feature importance analysis
- Confusion matrix plots

To run the notebook:
```bash
jupyter notebook notebook/data_analysis.ipynb
```

## Python Code Structure

The `python_code/` directory contains the core implementation:

1. **Neural Network Implementation** (`neural_network.py`):
   - Deep learning-based sentiment classifier
   - Support for different word embeddings (Word2Vec, GloVe)
   - Custom embedding layer
   - Dropout and regularization for improved generalization
   - Model training and evaluation
   - Prediction utilities

2. **Naive Bayes Implementation** (`naive_bayes.py`):
   - TF-IDF vectorization
   - Multinomial Naive Bayes classifier
   - Feature importance analysis
   - Performance metrics

3. **Word Embedding Utilities** (`embedding_utils.py`, `embedding_comparison.py`):
   - Word2Vec implementation for creating word embeddings
   - GloVe embeddings integration
   - Embedding comparison tools
   - Semantic similarity analysis
   - These embeddings can be used by both the Naive Bayes and Neural Network models

4. **Testing and Comparison** (`nn_test.py`, `comparison.py`):
   - Model testing utilities
   - Performance comparison tools
   - Cross-validation implementation

## Data Format

The input data should be a CSV file with the following columns:
- `column`: Contains either "positive" or "negative"
- `text`: Contains the review text

Example:
```csv
column,text
positive,"Great product, highly recommended!"
negative,"Not worth the money, disappointed."
```

## Usage

### Setting up the Environment

```bash
# Create virtual environment
python3.12 -m venv myenv

# Activate virtual environment
source myenv/bin/activate  # On Unix/macOS
myenv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Models

1. **Naive Bayes Classifier**:
```python
from python_code.naive_bayes import NaiveBayesSentimentClassifier

# Initialize and train
classifier = NaiveBayesSentimentClassifier()
df = classifier.load_data('data/reviews.csv')
X_train, X_test, y_train, y_test = classifier.prepare_data(df)
classifier.train(X_train, y_train)

# Make predictions
text = "This product is amazing!"
prediction, probability = classifier.predict(text)

# Evaluate model
metrics, report, cm = classifier.evaluate(X_test, y_test)
```

2. **Neural Network Classifier**:
```python
from python_code.neural_network import SentimentClassifier

# Initialize and train
classifier = SentimentClassifier()
classifier.load_data('data/reviews.csv')
classifier.train()

# Make predictions
text = "This product is amazing!"
prediction, probability = classifier.predict(text)

# Evaluate model
metrics = classifier.evaluate()
```

3. **Word Embedding Utilities**:
```python
from python_code.embedding_utils import WordEmbeddingUtils

# Initialize word embedding utilities
embedding_utils = WordEmbeddingUtils()

# Create Word2Vec embeddings
word2vec_model = embedding_utils.train_word2vec('data/reviews.csv')

# Load GloVe embeddings
glove_embeddings = embedding_utils.load_glove('glove.6B/glove.6B.100d.txt')

# Compare different embedding methods
comparison_results = embedding_utils.compare_embeddings(word2vec_model, glove_embeddings)
```

## Features

### Word Embeddings
- Word2Vec implementation for semantic word representations
- GloVe embeddings integration
- Embedding comparison and visualization tools
- Semantic similarity analysis

### Naive Bayes Classifier
- TF-IDF vectorization for text preprocessing
- Performance metrics (precision, recall, F1-score)
- Confusion matrix visualization
- Feature importance analysis
- Support for both single and batch predictions

### Neural Network (Deep Learning)
- Support for different word embeddings (Word2Vec, GloVe)
- Dropout layers for regularization
- Batch normalization for stable training
- Training history visualization
- Performance metrics comparison

## Model Performance

The system provides various metrics to evaluate model performance:
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curves (Neural Network)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Scikit-learn documentation
- TensorFlow documentation
- Natural Language Processing with Python