# Sentiment Analysis Classifier

This project implements two different approaches for sentiment analysis: a Naive Bayes classifier and a Deep Learning LSTM model. The system analyzes text reviews and classifies them as either positive or negative sentiments.

## Project Structure

```
Sentiment_Analysis_Classifier/
├── README.md
├── requirements.txt
├── naive_bayes_classifier.py
├── neural_network_classifier.py
├── data/
│   └── reviews.csv
└── models/
    ├── saved_nb_model/
    └── saved_nn_model/
```

## Requirements

- Python 3.12
- scikit-learn
- TensorFlow
- pandas
- numpy
- matplotlib
- seaborn

Install dependencies using:
```bash
pip install -r requirements.txt
```

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

### Running the Naive Bayes Classifier

```python
from naive_bayes_classifier import NaiveBayesSentimentClassifier

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

### Running the Neural Network Classifier

```python
from neural_network_classifier import SentimentComparison

# Initialize and run comparison
model_comparison, history = main()
```

## Features

### Naive Bayes Classifier
- TF-IDF vectorization for text preprocessing
- Performance metrics (precision, recall, F1-score)
- Confusion matrix visualization
- Feature importance analysis
- Support for both single and batch predictions

### Neural Network (LSTM)
- Word embeddings for semantic understanding
- LSTM layers for sequence processing
- Dropout for regularization
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