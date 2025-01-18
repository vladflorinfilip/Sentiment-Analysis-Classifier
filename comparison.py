from naive_bayes_classifier import NaiveBayesSentimentClassifier

# Initialize and train
classifier = NaiveBayesSentimentClassifier()
df = classifier.load_data('reviews.csv')
X_train, X_test, y_train, y_test = classifier.prepare_data(df)
classifier.train(X_train, y_train)

# Make predictions
text = "This product is amazing!"
prediction, probability = classifier.predict(text)
print(f"Prediction: {'Positive' if prediction[0] == 1 else 'Negative'}")
print(f"Confidence: {max(probability[0]):.2f}")

# Get model performance
metrics, report, cm = classifier.evaluate(X_test, y_test)
print(report)

# View most predictive words
top_features = classifier.get_most_informative_features()
print(top_features)