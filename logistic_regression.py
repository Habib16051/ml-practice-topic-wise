# Logistic regression is a statistical modeling technique used to predict the probability of a binary outcome
# (such as "yes" or "no," "success" or "failure") based on one or more independent variables,
# which can be continuous or categorical
import numpy as np
from sklearn.linear_model import LogisticRegression
# Example dataset: Hours studied (X) and Exam scores (Y)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)  # Studied hours
# Exam scores corresponding to the hours studied
Y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])  # Exam scores (0 or 1, binary classification)
# Create and train the logistic regression model
model = LogisticRegression()  # Instantiate the model
model.fit(X, Y)  # Train the model

# Function to predict exam score based on hours studied
def predict_exam_score(hours):
    return model.predict(np.array([[hours]]))[0]

# Example usage
if __name__ == "__main__":
    hours = 10
    result = predict_exam_score(hours)
    print(f"The predicted exam score for studying {hours} hours is {'Pass' if result == 1 else 'Fail'}")
    # Output: The predicted exam score for studying 6 hours is Pass
    # Output: The predicted exam score for studying 6 hours is Fail