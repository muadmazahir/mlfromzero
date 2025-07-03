from mlfromzero.models import Matrix, Vector


class LinearRegression:
    def __init__(self):
        self.weights: Vector = None
        self.bias: int = None

    def fit(self, input: list[list[int]], output: list[int]):
        """
        Fit the linear regression model to the data.

        Calculate weights and bias using the normal equation: θ = (X^T . X)^(-1) X^T . y

        Where:
        - X is the design matrix (features with bias term)
        - y is the target vector
        - θ is the parameter vector [bias, weights]
        """
        if not isinstance(input[0], list):
            raise TypeError("Input must be a list of lists")
        
        if not isinstance(output, list):
            raise TypeError("Output must be a list")

        
        # add 1 at the beginning of each row to represent bias
        biased_list = [[1] + sublist for sublist in input]

        input_matrix = Matrix(biased_list)

        # X^T
        input_matrix_transpose = input_matrix.transpose

        # X^T . X
        matrix_transpose_product = input_matrix_transpose * input_matrix

        # (X^T . X)^(-1)
        inverse_matrix_transpose_product = matrix_transpose_product.inverse

        # transpose output to a column vector
        output_transposed = Matrix([output]).transpose

        # X^T . y
        input_matrix_transpose_output = input_matrix_transpose * output_transposed

        # (X^T . X)^(-1) . X^T . y
        theta = inverse_matrix_transpose_product * input_matrix_transpose_output

        self.weights = Vector([weight[0] for weight in theta.matrix_list[1:]])
        self.bias = theta.matrix_list[0][0]

    def predict(self, input: list[int]) -> int:
        """
        Predict the output for a given input.

        Parameters:
        - input: list of input features

        Returns:
        - predicted output
        """
        return (self.weights * Vector(input)) + self.bias
