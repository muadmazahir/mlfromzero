import copy
import random

from mlfromzero.models import ActivationFunctions, Matrix, Vector
from mlfromzero.utils import MathsUtils


class Neuron:
    """
    A single neuron in a neural network layer.

    Each neuron maintains weights, a bias term, and an activation value.
    Neurons in the input layer have fixed weights of 1 and bias of 0 which do not change.
    """

    def __init__(self, num_inputs: int, input_layer: bool = False):
        """
        Initialize a neuron with random/fixed weights and bias.

        Args:
            num_inputs: Number of input connections to this neuron
            input_layer: If True, sets weights to 1 and bias to 0 (for input layer neurons)
        """
        if input_layer:
            self.weights = [1 for _ in range(num_inputs)]
            self.bias = 0
        else:
            self.weights = [random.random() for _ in range(num_inputs)]
            self.bias = random.random()
        self.activation_value = 0

    def __str__(self) -> str:
        return f"Neuron(weights={self.weights}, bias={self.bias}, activation_value={self.activation_value})"

    def __repr__(self) -> str:
        return f"weights={self.weights}, bias={self.bias}, activation_value={self.activation_value}"


class Layer:
    def __init__(self, neurons: list[Neuron], activation_function: str = "relu", input_layer: bool = False):
        self.neurons: list[Neuron] = neurons
        self.activation_function: str = activation_function
        self.input_layer: bool = input_layer

    def __len__(self) -> int:
        return len(self.neurons)

    def __getitem__(self, index: int) -> Neuron:
        return self.neurons[index]

    def __setitem__(self, index: int, value: Neuron):
        self.neurons[index] = value

    def __delitem__(self, index: int):
        del self.neurons[index]

    def __iter__(self):
        return iter(self.neurons)

    def __str__(self) -> str:
        return f"Layer(neurons={self.neurons}, activation_function={self.activation_function}, input_layer={self.input_layer})"

    def __repr__(self) -> str:
        return f"Layer(neurons={self.neurons}, activation_function={self.activation_function}, input_layer={self.input_layer})"


class FeedForwardNeuralNetwork:
    """
    A feedforward neural network implementation with backpropagation training.

    This class implements a multi-layer perceptron with configurable layers,
    activation function (currently only ReLU is supported), and gradient descent optimization.
    """

    def __init__(self):
        """Initialize an empty neural network."""
        self.input_layer_set = False
        self.layers: list[Layer] = []

    def add_layer(self, num_neurons, activation_function: str = "relu", input_layer: bool = False):
        """
        Add a layer to the neural network.

        Args:
            num_neurons: Number of neurons in the layer
            activation_function: Type of activation function (currently only `relu` is supported)
            input_layer: If True, creates an input layer with fixed weights

        Raises:
            ValueError: If trying to add a non-input layer before setting the input layer
        """
        if input_layer:
            self.layers.append(
                Layer(
                    neurons=[Neuron(num_inputs=1, input_layer=True) for _ in range(num_neurons)],
                    activation_function=activation_function,
                    input_layer=True,
                )
            )
            self.input_layer_set = True
        else:
            if not self.input_layer_set:
                raise ValueError("First set an input layer")

            # find the number of neurons in the last layer
            last_layer_neuron_num = len(self.layers[-1])

            self.layers.append(
                Layer(
                    neurons=[Neuron(num_inputs=last_layer_neuron_num, input_layer=False) for _ in range(num_neurons)],
                    activation_function=activation_function,
                    input_layer=False,
                )
            )

    def fit(self, inputs: list[list[int]], outputs: list[int], epochs: int = 3):
        """
        Train the neural network using backpropagation.

        Args:
            inputs: List of input samples, where each sample is a list of feature values
            outputs: List of target values corresponding to each input sample
            epochs: Number of training epochs

        Raises:
            ValueError: If input dimensions don't match the number of neurons in the first layer
        """
        if len(inputs[0]) != len(self.layers[0]):
            raise ValueError("Each input list should have the same number of values as the first layer")
        predicted_outputs = []
        for epoch in range(epochs):
            for input, output in zip(inputs, outputs):
                for layer in self.layers:
                    layer_activation_values_list = []
                    for neuron in layer:
                        layer_activation_values_list.append(self.forward(neuron, input, activation_function="relu"))
                    input = layer_activation_values_list  # previous layer's activation values are the input for the next layer
                predicted_output = layer_activation_values_list[0]  # last layer's activation value is the predicted output
                self.back_propagation(predicted_output=predicted_output, actual_output=output)
                predicted_outputs.append(predicted_output)
            cost = MathsUtils.caclulate_mean_squared_error(predicted_outputs, outputs)
            accuracy = MathsUtils.caclulate_accuracy(predicted_outputs, outputs)
            print(f"Epoch {epoch + 1} - Average Cost: {cost}, Accuracy: {accuracy}")

    def forward(self, neuron: Neuron, input: list[int], activation_function: str = "relu") -> float:
        """
        Perform forward propagation for a single neuron.

        Computes the weighted sum of inputs plus bias, then applies the activation function.

        Args:
            neuron: The neuron to process
            input: Input values to the neuron
            activation_function: Type of activation function (currently only `relu` is supported)

        Returns:
            float: The activation value after applying the activation function

        Raises:
            ValueError: If an unsupported activation function is specified
        """
        activation_value = (Vector(neuron.weights) * Vector(input)) + neuron.bias
        if activation_function == "relu":
            activation_value = ActivationFunctions.relu(activation_value)
            neuron.activation_value = activation_value
            return activation_value
        else:
            raise ValueError("Currently don't support any other type of activation functions")

    def back_propagation(self, predicted_output: int | float, actual_output: int | float, learning_rate: float = 0.001):
        """
        Perform backpropagation to update network weights and biases.

        This method implements gradient descent by:
        1. Computing the error signal: -2 * (actual_output - predicted_output)
        2. Iterating through layers (excluding the input layer)
        3. Computing partial derivatives for each weight and bias with respect to the loss function, which is Mean Squared Error
        4. Updating parameters using the gradient descent rule:
            new parameter = current parameter - (partial_derivative * learning_rate)

        The partial derivative for each weight is:
            (error signal) * (product of relevant weights in later layers) * (activation function derivative) *
            (activation of previous neuron)

        The partial derivative for each bias is:
            (error signal) * (product of relevant weights in later layers) * (activation function derivative)

        The partial derivative for the activation function is:
            1 if x > 0, 0 if x <= 0

        Args:
            predicted_output: A predicted output value of the network
            actual_output: The actual target output
            learning_rate: Step size for gradient descent (default: 0.001)
        """
        # take a copy of the layers so that we can take partial derivatives
        layers_copy = copy.deepcopy(self.layers)

        # The error signal which will be sued to calculate partial derivatives of parameters
        error = -2 * (actual_output - predicted_output)

        # reversed_layers = list(reversed(self.layers))
        for layer_index, (layer, layer_copy) in enumerate(zip(self.layers, layers_copy)):
            if not layer_index == 0:  # skip first layer since it is the input layer
                for neuron_index, (neuron, neuron_copy) in enumerate(zip(layer, layer_copy)):
                    if layer.activation_function == "relu":
                        activation_function_derivative = ActivationFunctions.relu_derivative(neuron.activation_value)
                    else:
                        raise ValueError("Currently don't support any other type of activation functions")
                    relevant_weights_multiplied = None

                    # Do matrix mulitplication of all weights of all neurons up to the layer before the current neuron's layer
                    # From the layer before the current neurons layer, mulitply only the weights of the current neuron's index
                    reverse_weights_list = list(reversed(self.layers[layer_index + 1 :]))

                    for reversed_layer in reverse_weights_list:
                        layer_weights_matrix = []
                        for n in reversed_layer:
                            if reversed_layer == reverse_weights_list[-1]:
                                layer_weights_matrix.append([n.weights[neuron_index]])
                            else:
                                layer_weights_matrix.append(n.weights)
                        if relevant_weights_multiplied is None:
                            relevant_weights_multiplied = Matrix(layer_weights_matrix)
                        else:
                            # multiply new weight matrix to the existing matrix from previous multiplications
                            relevant_weights_multiplied *= Matrix(layer_weights_matrix)
                    if relevant_weights_multiplied:
                        relevant_weights_multiplied_value = relevant_weights_multiplied[0][0]
                    else:
                        relevant_weights_multiplied_value = 1

                    for weight_index, weight in enumerate(neuron.weights):
                        previous_layer_neuron_activation_value = self.layers[layer_index - 1][
                            weight_index
                        ].activation_value  # get the corresponding previous neuron's activation value

                        weight_partial_derivative = (
                            error
                            * relevant_weights_multiplied_value
                            * activation_function_derivative
                            * previous_layer_neuron_activation_value
                        )
                        # update neuron in layers copy with new weight
                        neuron_copy.weights[weight_index] = self.calculate_new_parameter_value(
                            current_value=weight, partial_derivative=weight_partial_derivative, learning_rate=learning_rate
                        )

                    bias_partial_derivative = error * relevant_weights_multiplied_value * activation_function_derivative

                    # update neuron in layers copy with new bias
                    neuron_copy.bias = self.calculate_new_parameter_value(
                        current_value=neuron.bias, partial_derivative=bias_partial_derivative, learning_rate=learning_rate
                    )

        # assign the layers with the updated values to the networks layers.
        self.layers = layers_copy

    @staticmethod
    def calculate_new_parameter_value(
        current_value: int | float, partial_derivative: int | float, learning_rate: int | float = 0.001
    ):
        """
        Computes the new parameter value by subtracting the gradient step
        from the current value:
        new_value = current_value - (partial_derivative * learning_rate)

        Args:
            current_value: Current parameter value (weight or bias)
            partial_derivative: Partial derivative of this parameter with respect to the loss function
            learning_rate: Step size for gradient descent

        Returns:
            float: Updated parameter value
        """
        # calculate step size by multiplying partial derivative with learning rate
        step_size = partial_derivative * learning_rate

        # calculate new parameter value by subtracting step size from existing parameter value
        return current_value - step_size
