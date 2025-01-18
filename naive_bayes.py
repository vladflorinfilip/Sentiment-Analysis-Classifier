import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

class NaiveBayesSentimentClassifier:
    def __init__(self, max_features=10000):
        """
        Initialize the Naive Bayes sentiment classifier
        
        Parameters:
        max_features (int): Maximum number of features to use in TF-IDF vectorization
        """
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.classifier = MultinomialNB()
        self.is_trained = False

    def load_data(self, csv_file):
        """
        Load data from CSV file
        
        Parameters:
        csv_file (str): Path to CSV file with 'column' and 'text' columns
        """
        df = pd.read_csv(csv_file)
        df['sentiment'] = (df['column'] == 'positive').astype(int)
        return df

    def prepare_data(self, df, test_size=0.2, random_state=42):
        """
        Split data into training and test sets
        
        Parameters:
        df (pandas.DataFrame): DataFrame with 'text' and 'sentiment' columns
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        """
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'], 
            df['sentiment'],
            test_size=test_size,
            random_state=random_state
        )
        
        # Transform text data to TF-IDF features
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        return X_train_tfidf, X_test_tfidf, y_train, y_test

    def train(self, X_train, y_train):
        """
        Train the Naive Bayes classifier
        
        Parameters:
        X_train: Training features (TF-IDF matrix)
        y_train: Training labels
        """
        self.classifier.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, text):
        """
        Make predictions on new text
        
        Parameters:
        text (str or list): Text(s) to classify
        """
        if not self.is_trained:
            raise ValueError("Model needs to be trained before making predictions")
        
        # Handle both single strings and lists of strings
        if isinstance(text, str):
            text = [text]
        
        # Transform text using the fitted vectorizer
        text_tfidf = self.vectorizer.transform(text)
        predictions = self.classifier.predict(text_tfidf)
        probabilities = self.classifier.predict_proba(text_tfidf)
        
        return predictions, probabilities

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model's performance
        
        Parameters:
        X_test: Test features (TF-IDF matrix)
        y_test: True labels
        """
        if not self.is_trained:
            raise ValueError("Model needs to be trained before evaluation")
        
        predictions = self.classifier.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'precision': precision_score(y_test, predictions),
            'recall': recall_score(y_test, predictions),
            'f1': f1_score(y_test, predictions)
        }
        
        # Generate detailed classification report
        report = classification_report(y_test, predictions)
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, predictions)
        
        return metrics, report, cm

    def plot_confusion_matrix(self, cm):
        """
        Plot confusion matrix
        
        Parameters:
        cm: Confusion matrix from evaluate()
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    def get_most_informative_features(self, n=20):
        """
        Get the most informative features (words) for classification
        
        Parameters:
        n (int): Number of top features to return
        """
        if not self.is_trained:
            raise ValueError("Model needs to be trained before extracting features")
        
        # Get feature names and their coefficients
        feature_names = self.vectorizer.get_feature_names_out()
        coefs = self.classifier.coef_[0]
        
        # Create DataFrame of features and their coefficients
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefs
        })
        
        # Sort by absolute value of coefficient
        feature_importance['abs_coef'] = abs(feature_importance['coefficient'])
        feature_importance = feature_importance.sort_values('abs_coef', ascending=False)
        
        return feature_importance.head(n)

def main():
    # Initialize classifier
    classifier = NaiveBayesSentimentClassifier()
    
    # Load and prepare data
    df = classifier.load_data('reviews.csv')
    X_train, X_test, y_train, y_test = classifier.prepare_data(df)
    
    # Train the model
    classifier.train(X_train, y_train)
    
    # Evaluate
    metrics, report, cm = classifier.evaluate(X_test, y_test)
    
    # Print results
    print("\nClassification Report:")
    print(report)
    print("\nMetrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot confusion matrix
    classifier.plot_confusion_matrix(cm)
    
    # Show most informative features
    top_features = classifier.get_most_informative_features()
    print("\nMost Informative Features:")
    print(top_features)
    
    return classifier

if __name__ == "__main__":
    classifier = main()