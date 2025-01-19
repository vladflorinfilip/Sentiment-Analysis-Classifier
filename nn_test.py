# Save the code I provided earlier in 'sentiment_classifier.py'
from neural_network import main
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# Train the model
model , tokenizer, history = main()

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

# Test with some reviews
test_reviews = [
    "This movie was absolutely fantastic!",
    "In Black Mask, Jet Li plays a bio-engineered super-killer turned pacifist, who has to fight against other super-killers. Bad plot, bad sfx(60 million dollar budget), but the fighting scenes were excellent! Jet Li is the greatest martial-arts star alive!",
    "Hated this movies, so bad, awful movie",
    "Never watching this again, horrible",
    "I just saw the movie on tv. I really enjoyed it. I like a good mystery. and this one had me guessing up to the end. Sean Connery did a good job. I would recomend it to someone.",
    "I loved this movie. Captain America has had 3 movies in the past, not counting some 1940s ones which can be excused for being good, that were so awful they make movies like Steel or Green Lantern look good. However, I saw a lot of critiques for this movie that were negative. Now I understand differing opinions and all, but they all followed a formula and each one seemed like griping to the point it was just getting stupid. If you'll indulge me here, I have a list of the repeated ones and a rebuttal to each."
]

for review in test_reviews:
    result = predict_sentiment(review, model, tokenizer)
    print(f"\nReview: {review}")
    print(result)

# Save the model
model.save('sentiment_model.keras')
tokenizer_json = tokenizer.to_json()
with open('tokenizer.json', 'w') as f:
    f.write(tokenizer_json)

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