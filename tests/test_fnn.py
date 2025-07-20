import pytest

from mlfromzero.fnn import FeedForwardNeuralNetwork, Neuron
from mlfromzero.utils import MathsUtils


def test_neuron_initialization():
    """Test Neuron initialization with different parameters."""
    # Test input layer neuron
    input_neuron = Neuron(num_inputs=3, input_layer=True)
    assert input_neuron.weights == [1, 1, 1]
    assert input_neuron.bias == 0
    assert input_neuron.activation_value == 0

    # Test hidden/output layer neuron
    hidden_neuron = Neuron(num_inputs=2, input_layer=False)
    assert len(hidden_neuron.weights) == 2
    assert isinstance(hidden_neuron.bias, float)
    assert hidden_neuron.activation_value == 0


def test_neuron_string_representation():
    """Test Neuron string representation methods."""
    neuron = Neuron(num_inputs=2, input_layer=False)
    neuron.weights = [1, 2]
    neuron.bias = 3
    neuron.activation_value = 5

    assert "weights=[1, 2]" in str(neuron)
    assert "bias=3" in str(neuron)
    assert "activation_value=5" in str(neuron)


def test_fnn_initialization():
    """Test FeedForwardNeuralNetwork initialization."""
    fnn = FeedForwardNeuralNetwork()
    assert not fnn.input_layer_set
    assert fnn.layers == []


def test_add_input_layer():
    """Test adding input layer to the neural network."""
    fnn = FeedForwardNeuralNetwork()
    fnn.add_layer(num_neurons=3, input_layer=True)

    assert fnn.input_layer_set
    assert len(fnn.layers) == 1
    assert len(fnn.layers[0]) == 3

    # Check that input layer neurons have weights=1 and bias=0
    for neuron in fnn.layers[0]:
        assert neuron.weights == [1]
        assert neuron.bias == 0


def test_add_hidden_layer_without_input_layer():
    """Test that adding a hidden layer without input layer raises error."""
    fnn = FeedForwardNeuralNetwork()

    with pytest.raises(ValueError, match="First set an input layer"):
        fnn.add_layer(num_neurons=2, input_layer=False)


def test_add_hidden_layer():
    """Test adding hidden layer after input layer."""
    fnn = FeedForwardNeuralNetwork()
    fnn.add_layer(num_neurons=3, input_layer=True)
    fnn.add_layer(num_neurons=2, input_layer=False)

    assert len(fnn.layers) == 2
    assert len(fnn.layers[1]) == 2

    # Check that hidden layer neurons have correct number of inputs
    for neuron in fnn.layers[1]:
        assert len(neuron.weights) == 3  # Should match number of neurons in previous layer


def test_add_multiple_layers():
    """Test adding multiple layers to the neural network."""
    fnn = FeedForwardNeuralNetwork()
    fnn.add_layer(num_neurons=2, input_layer=True)  # Input layer: 2 neurons
    fnn.add_layer(num_neurons=3, input_layer=False)  # Hidden layer: 3 neurons
    fnn.add_layer(num_neurons=1, input_layer=False)  # Output layer: 1 neuron

    assert len(fnn.layers) == 3
    assert len(fnn.layers[0]) == 2  # Input layer
    assert len(fnn.layers[1]) == 3  # Hidden layer
    assert len(fnn.layers[2]) == 1  # Output layer

    # Check weights dimensions
    for neuron in fnn.layers[1]:  # Hidden layer neurons
        assert len(neuron.weights) == 2  # Should match input layer size

    for neuron in fnn.layers[2]:  # Output layer neurons
        assert len(neuron.weights) == 3  # Should match hidden layer size


def test_forward_pass_relu():
    """Test forward pass with ReLU activation function."""
    fnn = FeedForwardNeuralNetwork()

    # Create a simple network: 2 inputs -> 1 output
    fnn.add_layer(num_neurons=2, input_layer=True)
    fnn.add_layer(num_neurons=1, input_layer=False)

    # Set specific weights for testing
    fnn.layers[1][0].weights = [1, 2]  # Output neuron weights
    fnn.layers[1][0].bias = 1  # Output neuron bias

    # Test ReLU with positive activation
    result = fnn.forward(fnn.layers[1][0], [1, 1], "relu")
    expected = max(0, (1 * 1 + 2 * 1) + 1)  # max(0, 4) = 4
    assert result == expected
    assert fnn.layers[1][0].activation_value == expected

    # Test ReLU with negative activation
    fnn.layers[1][0].weights = [-1, -2]
    fnn.layers[1][0].bias = -1
    result = fnn.forward(fnn.layers[1][0], [1, 1], "relu")
    expected = max(0, (-1 * 1 + -2 * 1) + -1)  # max(0, -4) = 0
    assert result == expected


def test_forward_pass_unsupported_activation():
    """Test that unsupported activation functions raise error."""
    fnn = FeedForwardNeuralNetwork()
    fnn.add_layer(num_neurons=1, input_layer=True)
    fnn.add_layer(num_neurons=1, input_layer=False)

    with pytest.raises(ValueError, match="Currently don't support any other type of activation functions"):
        fnn.forward(fnn.layers[1][0], [1], "sigmoid")


def test_calculate_new_parameter_value():
    """Test parameter update calculation."""
    fnn = FeedForwardNeuralNetwork()

    # Test with positive partial derivative
    new_value = fnn.calculate_new_parameter_value(current_value=10.0, partial_derivative=2.0, learning_rate=0.1)
    expected = 10.0 - (2.0 * 0.1)  # 10.0 - 0.2 = 9.8
    assert new_value == expected

    # Test with negative partial derivative
    new_value = fnn.calculate_new_parameter_value(current_value=5.0, partial_derivative=-1.0, learning_rate=0.01)
    expected = 5.0 - (-1.0 * 0.01)  # 5.0 + 0.01 = 5.01
    assert new_value == expected


def test_fit_input_output_mismatch():
    """Test that fit raises error when input dimensions don't match."""
    fnn = FeedForwardNeuralNetwork()
    fnn.add_layer(num_neurons=2, input_layer=True)
    fnn.add_layer(num_neurons=1, input_layer=False)

    inputs = [[1, 2, 3]]  # 3 features but network expects 2
    outputs = [1]

    with pytest.raises(ValueError, match="Each input list should have the same number of values as the first layer"):
        fnn.fit(inputs, outputs, epochs=1)


def test_simple_training():
    """Test simple training scenario."""
    fnn = FeedForwardNeuralNetwork()
    fnn.add_layer(num_neurons=1, input_layer=True)
    fnn.add_layer(num_neurons=1, input_layer=False)

    # Simple XOR-like problem
    inputs = [[0], [1]]
    outputs = [0, 1]

    # This should run without errors
    fnn.fit(inputs, outputs, epochs=1)


def test_back_propagation_simple():
    """Test back propagation with a simple network."""
    fnn = FeedForwardNeuralNetwork()
    fnn.add_layer(num_neurons=1, input_layer=True)
    fnn.add_layer(num_neurons=1, input_layer=False)

    # Set initial weights and bias
    fnn.layers[1][0].weights = [1.0]
    fnn.layers[1][0].bias = 0.0

    # Store original values
    original_weight = fnn.layers[1][0].weights[0]
    original_bias = fnn.layers[1][0].bias

    # Set input layer neuron activation value
    fnn.layers[0][0].activation_value = 1.0

    # Perform forward pass to set output layer activation values
    fnn.forward(fnn.layers[1][0], [1.0], "relu")

    # Run back propagation
    fnn.back_propagation(predicted_output=0.5, actual_output=1.0, learning_rate=0.1)

    # Check that weights and bias were updated
    assert fnn.layers[1][0].weights[0] != original_weight
    assert fnn.layers[1][0].bias != original_bias


def test_utils_functions():
    """Test utility functions used by the neural network."""
    # Test mean squared error
    predicted = [1, 2, 3]
    actual = [1, 2, 4]
    mse = MathsUtils.caclulate_mean_squared_error(predicted, actual)
    expected_mse = ((1 - 1) ** 2 + (2 - 2) ** 2 + (3 - 4) ** 2) / 3  # 1/3
    assert mse == expected_mse

    # Test accuracy
    predicted = [1, 0, 1, 0]
    actual = [1, 0, 0, 0]
    accuracy = MathsUtils.caclulate_accuracy(predicted, actual)
    expected_accuracy = (3 / 4) * 100  # 75%
    assert accuracy == expected_accuracy


def test_complex_network_training():
    """Test training with a more complex network architecture."""
    fnn = FeedForwardNeuralNetwork()
    fnn.add_layer(num_neurons=2, input_layer=True)
    fnn.add_layer(num_neurons=3, input_layer=False)
    fnn.add_layer(num_neurons=1, input_layer=False)

    # Training data
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    outputs = [0, 1, 1, 0]  # XOR problem

    # This should run without errors
    fnn.fit(inputs, outputs, epochs=2)


def test_network_architecture_validation():
    """Test that network architecture is properly validated."""
    fnn = FeedForwardNeuralNetwork()

    # Test adding layers in correct order
    fnn.add_layer(num_neurons=2, input_layer=True)
    fnn.add_layer(num_neurons=3, input_layer=False)
    fnn.add_layer(num_neurons=1, input_layer=False)

    # Verify layer connections
    assert len(fnn.layers[0]) == 2  # Input layer
    assert len(fnn.layers[1]) == 3  # Hidden layer
    assert len(fnn.layers[2]) == 1  # Output layer

    # Verify weight dimensions
    for neuron in fnn.layers[1]:  # Hidden layer
        assert len(neuron.weights) == 2  # Connected to input layer

    for neuron in fnn.layers[2]:  # Output layer
        assert len(neuron.weights) == 3  # Connected to hidden layer


def test_neuron_activation_value_persistence():
    """Test that neuron activation values persist after forward pass."""
    fnn = FeedForwardNeuralNetwork()
    fnn.add_layer(num_neurons=1, input_layer=True)
    fnn.add_layer(num_neurons=1, input_layer=False)

    # Set initial activation value
    fnn.layers[1][0].activation_value = 0.0

    # Run forward pass
    result = fnn.forward(fnn.layers[1][0], [1], "relu")

    # Check that activation value was updated
    assert fnn.layers[1][0].activation_value == result
    assert fnn.layers[1][0].activation_value != 0.0


def test_learning_rate_impact():
    """Test that different learning rates affect parameter updates differently."""
    fnn = FeedForwardNeuralNetwork()
    fnn.add_layer(num_neurons=1, input_layer=True)
    fnn.add_layer(num_neurons=1, input_layer=False)

    # Set initial weights
    fnn.layers[1][0].weights = [1.0]
    fnn.layers[1][0].bias = 0.0

    # Store original values
    original_weight = fnn.layers[1][0].weights[0]
    original_bias = fnn.layers[1][0].bias

    # Set input layer neuron activation value (this is what the input would be)
    fnn.layers[0][0].activation_value = 1.0

    # Perform forward pass to set output layer activation values
    fnn.forward(fnn.layers[1][0], [1.0], "relu")

    # Test with high learning rate
    fnn.back_propagation(predicted_output=0.5, actual_output=1.0, learning_rate=0.5)
    high_lr_weight_change = abs(fnn.layers[1][0].weights[0] - original_weight)
    high_lr_bias_change = abs(fnn.layers[1][0].bias - original_bias)

    # Reset and test with low learning rate
    fnn.layers[1][0].weights = [1.0]
    fnn.layers[1][0].bias = 0.0

    # Set input layer neuron activation value again
    fnn.layers[0][0].activation_value = 1.0

    # Perform forward pass again to set output layer activation values
    fnn.forward(fnn.layers[1][0], [1.0], "relu")

    fnn.back_propagation(predicted_output=0.5, actual_output=1.0, learning_rate=0.01)
    low_lr_weight_change = abs(fnn.layers[1][0].weights[0] - original_weight)
    low_lr_bias_change = abs(fnn.layers[1][0].bias - original_bias)

    # Higher learning rate should cause larger changes
    assert high_lr_weight_change > low_lr_weight_change
    assert high_lr_bias_change > low_lr_bias_change
