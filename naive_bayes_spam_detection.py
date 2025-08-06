from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Use more descriptive variable names and consistent label encoding
messages = [
    "Buy cheap meds now",
    "Limited time offer",
    "Get rich quick",
    "Hello, how are you?"
]
labels = [1, 1, 1, 0]  # 1: spam, 0: not spam

# Initialize CountVectorizer and fit_transform in one step
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(messages)

# Train the Naive Bayes classifier
model = MultinomialNB()
model.fit(X, labels)


def predict_spam(message: str) -> str:
    """Predict if a message is spam or not spam."""
    message_vector = vectorizer.transform([message])
    prediction = model.predict(message_vector)[0]
    return "Spam" if prediction == 1 else "Not Spam"


# Example usage
if __name__ == "__main__":
    test_message = "Get rich quickly"
    result = predict_spam(test_message)
    print(f"The message '{test_message}' is classified as: {result}")
    # Output: The message 'Get your free meds now' is classified as: Spam
