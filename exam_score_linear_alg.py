from sklearn.linear_model import LinearRegression
import numpy as np

# Dataset: Exam scores and hours studied
hours_studied = np.array([[1], [2], [3], [4], [5]])  # Hours studied X
exam_scores = np.array([3, 5, 7, 9, 11])  # Exam scores Y

# Equation: Exam Score = 2 * Hours Studied + 1
# This means for every hour studied, the exam score increases by 2 points, starting from a base score of 1 point.
# Y = 2 * X + 1

# Train the linear regression model
model = LinearRegression()
model.fit(hours_studied, exam_scores)


# Function to predict exam score based on hours studied
def predict_exam_score(hours):
    hours = np.array([[hours]])
    return model.predict(hours)[0]



# Example usage
if __name__ == "__main__":
    hours = 6  # Example hours studied
    predicted_score = predict_exam_score(hours)
    print(f"The predicted exam score for studying {hours} hours is {predicted_score:.2f}")