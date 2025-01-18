# Save the code I provided earlier in 'sentiment_classifier.py'
from neural_network import main

# Train the model
model, tokenizer, history = main()

# Now you can make predictions on new reviews
def predict_sentiment(text, model, tokenizer):
    # Preprocess new text
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=200)
    
    # Get prediction
    prediction = model.predict(padded)[0][0]
    confidence = max(prediction, 1 - prediction)
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    
    return f"Sentiment: {sentiment} (confidence: {confidence:.2f})"

# Try it out
new_review = "This movie was absolutely fantastic!"
result = predict_sentiment(new_review, model, tokenizer)
print(result)

# Save the model
model.save('sentiment_model')
tokenizer_json = tokenizer.to_json()
with open('tokenizer.json', 'w') as f:
    f.write(tokenizer_json)

# Later, load it back:
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json

model = load_model('sentiment_model')
with open('tokenizer.json') as f:
    tokenizer = tokenizer_from_json(f.read())

import matplotlib.pyplot as plt

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()