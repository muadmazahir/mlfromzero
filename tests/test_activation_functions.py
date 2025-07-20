from mlfromzero.models import ActivationFunctions


class TestActivationFunctions:
    """Test suite for the ActivationFunctions class."""

    def test_relu(self):
        """Test ReLU function with positive input values."""
        # Test various positive values
        assert ActivationFunctions.relu(5.0) == 5.0
        assert ActivationFunctions.relu(0.1) == 0.1
        assert ActivationFunctions.relu(100.0) == 100.0
        assert ActivationFunctions.relu(0.001) == 0.001

        assert ActivationFunctions.relu(-5.0) == 0.0
        assert ActivationFunctions.relu(-0.1) == 0.0
        assert ActivationFunctions.relu(-100.0) == 0.0
        assert ActivationFunctions.relu(-0.001) == 0.0

        assert ActivationFunctions.relu(0.0) == 0
        assert ActivationFunctions.relu(0) == 0

        assert ActivationFunctions.relu(5) == 5
        assert ActivationFunctions.relu(0) == 0

        assert ActivationFunctions.relu(-5) == 0

        assert ActivationFunctions.relu(5.0) == ActivationFunctions.relu(5)
        assert ActivationFunctions.relu(-5.0) == ActivationFunctions.relu(-5)
        assert ActivationFunctions.relu(0.0) == ActivationFunctions.relu(0)

    def test_relu_derivative_positive_values(self):
        """Test ReLU derivative function with positive input values."""
        # Test various positive values
        assert ActivationFunctions.relu_derivative(5.0) == 1.0
        assert ActivationFunctions.relu_derivative(-5.0) == 0.0

        assert ActivationFunctions.relu_derivative(0.0) == 0.0
        assert ActivationFunctions.relu_derivative(0) == 0.0

        # Positive integers
        assert ActivationFunctions.relu_derivative(5) == 1.0
        assert ActivationFunctions.relu_derivative(1) == 1.0

        # Negative integers
        assert ActivationFunctions.relu_derivative(-5) == 0.0
        assert ActivationFunctions.relu_derivative(-1) == 0.0
