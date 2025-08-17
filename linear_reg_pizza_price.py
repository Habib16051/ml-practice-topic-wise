from sklearn.linear_model import LinearRegression
import numpy as np

diameters = np.array([[6], [8], [10], [12], [14]])  # in inches
prices = np.array([8, 10, 12, 14, 16])  # in dollars

# Train Model
model = LinearRegression()
model.fit(diameters, prices)

# Predict Price
def predict_price(diameter):
    diameter = np.array([[diameter]])  # Reshape for single prediction
    return model.predict(diameter)[0]
# Example usage
if __name__ == "__main__":
    diameter = 19  # Example diameter in inches
    price = predict_price(diameter)
    print(f"The predicted price for a pizza with diameter {diameter} inches is ${price:.2f}")