class MathsUtils:
    """
    A utility class containing common mathematical operations used in ML algorithms.
    """

    @staticmethod
    def caclulate_mean_squared_error(predicted: list[int | float], actual: list[int | float]) -> int | float:
        """
        Calculate the mean squared error of the predicted and actual values.

        Args:
            predicted: List of predicted values
            actual: List of actual values

        Returns:
            float: The mean squared error

        MSE formula = 1/n * sum((predicted - actual) ** 2)
        """
        squared_error_sum = 0
        for predicted_value, actual_value in zip(predicted, actual):
            squared_error_sum += (predicted_value - actual_value) ** 2

        return squared_error_sum / len(predicted)

    @staticmethod
    def caclulate_accuracy(predicted: list[int | float], actual: list[int | float]) -> int | float:
        """
        Calculate the accuracy of the predicted and actual values.

        Args:
            predicted: List of predicted values
            actual: List of actual values

        Returns:
            float: The accuracy percentage (0-100)

        accuracy formula = (correct predictions / total predictions) * 100
        """
        correct_predictions = 0
        for predicted_value, actual_value in zip(predicted, actual):
            if predicted_value == actual_value:
                correct_predictions += 1

        return (correct_predictions / len(predicted)) * 100
