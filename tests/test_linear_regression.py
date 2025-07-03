import numpy as np
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

from mlfromzero.linear_regression import LinearRegression


def test_linear_regression():
    """
    Test the linear regression model using a simple celcius to fahrenheit conversion.
    """
    # Celsius values (input features)
    input = [[-40], [-20], [0], [10], [20], [30], [40], [50], [60], [70], [80], [90], [100]]

    # Corresponding Fahrenheit values (output targets)
    output = [-40, -4, 32, 50, 68, 86, 104, 122, 140, 158, 176, 194, 212]

    model = LinearRegression()
    model.fit(input, output)

    assert model.predict([25]) == 77

def test_linear_regression_with_multiple_features():
    """
    Test linear regression with multiple features by comparing with sklearn implementation.
    """
    # Generate 100 samples
    n_samples = 10

    # Feature 1: house size (1000-3000 sq ft)
    house_size = np.random.uniform(1000, 3000, n_samples)
    
    # Feature 2: number of bedrooms (1-5)
    bedrooms = np.random.randint(1, 6, n_samples)

    # Feature 3: age of house (0-30 years)
    age = np.random.uniform(0, 30, n_samples)
    
    # Create target: price = 100 + 0.1 * size + 20 * bedrooms - 2 * age + noise
    true_price = 100 + 0.1 * house_size + 20 * bedrooms - 2 * age
    noise = np.random.normal(0, 10, n_samples)
    price = true_price + noise

    # Convert input and output to lists
    input_data = [[size, bed, age_val] for size, bed, age_val in zip(house_size, bedrooms, age)]
    output_data = price.tolist()

    # Train your linear regression model
    model = LinearRegression()
    model.fit(input_data, output_data)

    # Train sklearn linear regression model
    sklearn_model = SklearnLinearRegression()
    sklearn_model.fit(input_data, output_data)

    # Test the models
    test_input = [2000, 3, 10]
    model_prediction = model.predict(test_input)
    sklearn_prediction = sklearn_model.predict([test_input])

    assert round(model_prediction, 2) == round(sklearn_prediction[0], 2)
