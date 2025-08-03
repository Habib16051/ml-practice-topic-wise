from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Example dataset: Movie genres based on feature
# Features: [Action, Comedy, Drama, Sci-Fi]

# Movie data

ratings = [8.0, 6.2, 7.2, 8.2]
durations = [160, 170, 168, 155]
features = np.column_stack((ratings, durations))
genres = [0, 0, 1, 1]  # 0: Action, 1: Comedy

# Initialize the KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(features, genres)

# Function to predict movie genre based on rating and duration
def predict_genre(rating, duration):
    feature = np.array([[rating, duration]])
    return knn.predict(feature)[0]

# Example usage
if __name__ == "__main__":
    rating = 7.4  # Example rating
    duration = 114  # Example duration in minutes
    genre = predict_genre(rating, duration)
    genre_name = "Action" if genre == 0 else "Comedy"
    print(f"The predicted genre for a movie with rating {rating} and duration {duration} minutes is {genre_name}.")
