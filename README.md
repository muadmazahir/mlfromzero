# ML From Zero 🚀

Implementation of machine learning algorithms from scratch using pure Python, without relying on third-party ML libraries. This project demonstrates the mathematical foundations and computational principles behind popular ML algorithms.

## 📚 Implemented Algorithms

- Linear Regression

### 🔄 Coming Soon
- Neural Networks
- Transformer

## Basic Usage

```python
from mlfromzero.linear_regression import LinearRegression

# Create sample data
X = [[1], [2], [3], [4], [5]]  # Features
y = [2, 4, 6, 8, 10]           # Targets

# Train the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
prediction = model.predict([6])
print(f"Prediction for input 6: {prediction}")  # Expected: 12
```


## 📊 Mathematical Implementation Details

### Linear Regression: Normal Equation

The linear regression implementation uses the normal equation for optimal parameter estimation:

**Formula**: θ = (X^T X)^(-1) X^T y

**Steps**:
1. **Design Matrix**: Add bias term (column of 1s) to feature matrix
2. **Matrix Operations**: 
   - Compute X^T (transpose)
   - Compute X^T X (matrix multiplication)
   - Compute (X^T X)^(-1) (matrix inverse)
   - Compute X^T y (matrix-vector multiplication)
3. **Parameter Extraction**: Extract bias and weights from θ vector

### Matrix and Vectors

All matrix and vector operations are written from scratch.


