from typing import Union


class Vector:
    def __init__(self, vector_list: list[int]):
        # Check if vector_list is actually a list
        if not isinstance(vector_list, list):
            raise TypeError(f"vector_list must be a list, got {type(vector_list)}")

        self.vector_list = vector_list

    def __add__(self, addend: 'Vector') -> 'Vector':
        """
        A vector added by a vector returns a vector: x + z = [x₁ + z₁, x₂ + z₂, x₃ + z₃, ..., xₙ + zₙ]
        """
        final_vector_list = []
        for index, feature in enumerate(self.vector_list):
            final_vector_list.append(feature + addend.vector_list[index])
        return Vector(final_vector_list)

    def __mul__(self, multiplier: Union['Vector', int]) -> Union['Vector', int]:
        """
        A vector multiplied by a scalar returns a vector: xc = [cx₁, cx₂, cx₃, ..., cxₙ]

        A vector multiplied by a vector returns a scalar: wx = w₁.x₁ + w₂.x₂ + w₃.x₃ ... wₙxₙ
        """
        if isinstance(multiplier, int):
            final_vector_list = []
            for feature in self.vector_list:
                final_vector_list.append(multiplier * feature)
            return Vector(final_vector_list)

        if isinstance(multiplier, Vector):
            final_value = 0
            for index, feature in enumerate(self.vector_list):
                final_value += feature * multiplier.vector_list[index]
            return final_value
        
        raise ValueError('multiplier should be either of type `Vector` or `int`')

    def __sub__(self, subtrahend: 'Vector') -> 'Vector':
        """
        A vector subtracted by a vector returns a vector: x - z = [x₁ - z₁, x₂ - z₂, x₃ - z₃, ..., xₙ - zₙ]
        """
        final_vector_list = []
        for index, feature in enumerate(self.vector_list):
            final_vector_list.append(feature - subtrahend.vector_list[index])
        return Vector(final_vector_list)

    def __len__(self):
        return len(self.vector_list)


class Matrix:
    def __init__(self, matrix_list: list[list[int]]):
        # Check if matrix_list is actually a list
        if not isinstance(matrix_list, list):
            raise TypeError(f"matrix_list must be a list, got {type(matrix_list)}")

        self.matrix_list: list[list[int]] = matrix_list
    
    @property
    def shape(self) -> tuple[int, int]:
        num_rows = len(self.matrix_list)
        num_columns = len(self.matrix_list[0]) if self.matrix_list else 0

        return num_rows, num_columns

    @property
    def transpose(self) -> 'Matrix':
        # Transpose using list comprehension
        return Matrix([[row[i] for row in self.matrix_list] for i in range(len(self.matrix_list[0]))])

    @property
    def inverse(self) -> 'Matrix':
        """
        Calculates the inverse of a matrix using the Gauss-Jordan elimination method.
        """
        n = len(self.matrix_list)
        augmented_matrix = [row[:] for row in self.matrix_list]
        inverse_matrix = self.identity(n).matrix_list

        for fd in range(n):
            fd_scaler = 1.0 / augmented_matrix[fd][fd]
            for j in range(n):
                augmented_matrix[fd][j] *= fd_scaler
                inverse_matrix[fd][j] *= fd_scaler
            for i in range(n):
                if i != fd:
                    cr_scaler = augmented_matrix[i][fd]
                    for j in range(n):
                        augmented_matrix[i][j] = augmented_matrix[i][j] - cr_scaler * augmented_matrix[fd][j]
                        inverse_matrix[i][j] = inverse_matrix[i][j] - cr_scaler * inverse_matrix[fd][j]
        return Matrix(inverse_matrix)

    @staticmethod
    def identity(size: int) -> 'Matrix':
        """
        Returns an identity matrix of size `size`
        """
        return Matrix([[1 if i == j else 0 for j in range(size)] for i in range(size)])

    def __add__(self, addend: 'Matrix') -> 'Matrix':
        if self.shape != addend.shape:
            raise ValueError('The shape of both matrices need to be the same to do addition')
        
        final_matrix = []

        for row, addend_row in zip(self.matrix_list, addend.matrix_list):
            row_additions = []
            for value, addend_value in zip(row, addend_row):
                row_additions.append(value + addend_value)
            final_matrix.append(row_additions)
        
        return Matrix(final_matrix)

    def __sub__(self, subtrahend: 'Matrix') -> 'Matrix':
        if self.shape != subtrahend.shape:
            raise ValueError('The shape of both matrices need to be the same to do addition')
        
        final_matrix = []

        for row, subtrahend_row in zip(self.matrix_list, subtrahend.matrix_list):
            row_subtractions = []
            for value, subtrahend_value in zip(row, subtrahend_row):
                row_subtractions.append(value - subtrahend_value)
            final_matrix.append(row_subtractions)
        
        return Matrix(final_matrix)

    def __mul__(self, multiplier: Union['Matrix', int]) -> 'Matrix':
        final_matrix = []
        if isinstance(multiplier, int):
            for row in self.matrix_list:
                row_multiplications = []
                for value in row:
                    row_multiplications.append(value * multiplier)
                final_matrix.append(row_multiplications)

        if isinstance(multiplier, Matrix):
            """
            A = [[1, 2], [3, 4]]
            B = [[5, 6], [7, 8]]

            A * B = [
                        [
                            (1 * 5) + (2 * 7) , (1 * 6) + (2 * 8)
                        ],
                        [
                            (3 * 5) + (4 * 7) , (3 * 6) + (4 * 8)
                        ]
                    ]

                =   [
                        [
                            19 , 22
                        ],
                        [
                            43, 50
                        ]
                    ] 
            """
            num_rows_A = len(self.matrix_list)
            num_cols_A = len(self.matrix_list[0])
            num_rows_B = len(multiplier.matrix_list)
            num_cols_B = len(multiplier.matrix_list[0])
            
            if num_cols_A != num_rows_B:
                raise ValueError("Number of rows of multiplier has to match number of columns of matrix you want to multiply it to")

            # Initialize result matrix with zeros
            final_matrix = [[0 for _ in range(num_cols_B)] for _ in range(num_rows_A)]
            
            # Perform multiplication
            for i in range(num_rows_A):
                for j in range(num_cols_B):
                    for k in range(num_cols_A):
                        final_matrix[i][j] += self.matrix_list[i][k] * multiplier.matrix_list[k][j]
            
        return Matrix(final_matrix)


def caclulate_mean_squared_error(predicted: list[int], actual: list[int]) -> int:
    """
    Calculate the mean squared error of the predicted and actual values.
    
    mean squared error formula = 1/n * sum((predicted - actual) ** 2)
    """
    squared_error_sum = 0
    for predicted_value, actual_value in zip(predicted, actual):
        squared_error_sum += (predicted_value - actual_value) ** 2
    
    return squared_error_sum / len(predicted)
