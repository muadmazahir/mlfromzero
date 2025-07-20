# ML From Zero 🚀

Implementation of machine learning algorithms from scratch using pure Python, without relying on third-party ML libraries. This project demonstrates the mathematical foundations and computational principles behind popular ML algorithms.

## 📚 Implemented Algorithms

- Linear Regression
- Feed Forward Neural Network (FNN)

### 🔄 Coming Soon
- Transformer

## Basic Usage

### Linear Regression

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

### Feed Forward Neural Network

```python
from mlfromzero.fnn import FeedForwardNeuralNetwork

# Create a neural network
nn = FeedForwardNeuralNetwork()

# Add layers (input layer must be added first)
nn.add_layer(num_neurons=2, activation_function="relu", input_layer=True)  # Input layer
nn.add_layer(num_neurons=3, activation_function="relu")  # Hidden layer
nn.add_layer(num_neurons=1, activation_function="relu")  # Output layer

# Training data
inputs = [[1, 2], [2, 3], [3, 4], [4, 5]]
outputs = [3, 5, 7, 9]  # Target values

# Train the network
nn.fit(inputs, outputs, epochs=100)
```

## 📊 Mathematical Implementation Details

### Linear Regression: Normal Equation

The linear regression implementation uses the normal equation for optimal parameter estimation:

**Formula**:
$\boldsymbol{\theta} = \left( \mathbf{X}^\top \mathbf{X} \right)^{-1} \mathbf{X}^\top \mathbf{y}$

**Steps**:
1. **Design Matrix**: Add bias term (column of 1s) to feature matrix
2. **Matrix Operations**: 
   - Compute $X^T$ (transpose)
   - Compute $X^T X$
   - Compute $(X^T X)^{-1}$ (matrix inverse)
   - Compute $X^Ty$
3. **Parameter Extraction**: Extract bias and weights from θ vector

### Feed Forward Neural Network: Backpropagation

The FNN implementation uses gradient descent with backpropagation for training:

**Forward Propagation**:
- **Weighted Sum**: $z = \sum_{i=1}^{n} w_i x_i + b $
- **Activation**: $a = ReLU(z) = max(0, z)$

**Backpropagation**:
- **Error Signal**: 

$$
\delta = -2 \cdot \text{actual\_output} - \text{predicted\_output}) 
$$

- **Weight Update**: $w_{\text{new}} = w_{\text{old}} - \left( \frac{\partial E}{\partial w} \cdot \text{learning\_rate} \right)$


- **Bias Update**: $b_{\text{new}} = b_{\text{old}}$ - $\left( \frac{\partial E}{\partial b} \cdot \text{learning\_rate} \right)$

**Partial Derivative of Parameters**:
- **Weight Gradient**:
$\frac{\partial E}{\partial w} = \delta \cdot \left( \prod_{i}^{n} w_i \right) \cdot \text{ReLU}'(z_i) \cdot a_{\text{prev}_{i-1}}$

- **Bias Gradient**: 
$\frac{\partial E}{\partial w} = \delta \cdot \left( \prod_{i}^{n} w_i \right) \cdot \text{ReLU}'(z_i)$

- **ReLU Derivative**: 

$$
\text{ReLU}(z) = \begin{cases}1 & \text{if } z > 0 \\ 0 & \text{if } z \leq 0\end{cases}
$$

**Note** - See [here](docs/partial_derivatives_calculation.md) for more details on how partial derivatives of parameters were calculated.

**Loss Function**:
$\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$

### Matrix and Vectors

All matrix and vector operations are written from scratch.
