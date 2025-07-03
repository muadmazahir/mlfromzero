import numpy as np

from mlfromzero.models import Matrix


def test_matrix_operations():
    matrix_1 =  Matrix([[1, 2], [3, 4]])
    matrix_2 = Matrix([[5, 6], [7, 8]])

    matrix_added = matrix_1 + matrix_2
    matrix_multiplied = matrix_1 * matrix_2
    matrix_multiplied_by_scalar = matrix_1 * 2
    matrix_subtracted = matrix_1 - matrix_2

    assert matrix_added.matrix_list == [[6, 8], [10, 12]]
    assert matrix_multiplied.matrix_list == [[19, 22], [43, 50]]
    assert matrix_multiplied_by_scalar.matrix_list == [[2, 4], [6, 8]]
    assert matrix_subtracted.matrix_list == [[-4, -4], [-4, -4]]

def test_matrix_multiplication_large_matrices():
    """
    This test is to verify that the matrix multiplication is working correctly for large matrices.

    First do matrix multiplication using numpy's matrix multiplication.
    Second do matrix multiplication using our own matrix multiplication.
    Third verify that the results are the same.
    """
    # Create two large matrices (e.g., 500 x 300 and 300 x 400)
    A = np.random.randint(0, 10, (4, 10))
    B = np.random.randint(0, 10, (10, 6))

    # Perform matrix multiplication
    C = np.dot(A, B)

    matrix_A =  Matrix(A.tolist())
    matrix_B = Matrix(B.tolist())

    matrix_C = matrix_A * matrix_B

    assert matrix_C.matrix_list == C.tolist()


def test_matrix_transpose():
    """
    Test the matrix transpose function
    """
    matrix =  Matrix([[1, 2], [3, 4]])
    matrix_transposed = matrix.transpose

    assert matrix_transposed.matrix_list == [[1, 3], [2, 4]]

def test_matrix_identity():
    """
    Test the matrix identity function
    """
    matrix =  Matrix([[1, 2], [3, 4]])
    matrix_identity = matrix.identity(matrix.shape[0])

    assert matrix_identity.matrix_list == [[1, 0], [0, 1]]

def test_matrix_inverse():
    """
    Test the matrix inverse function
    """
    matrix =  Matrix([[1, 2], [3, 4]])
    matrix_inverse = matrix.inverse

    assert matrix_inverse.matrix_list == [[-2, 1], [1.5, -0.5]]