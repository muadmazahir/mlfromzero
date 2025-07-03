from mlfromzero.models import Vector


def test_vector_operatios():
    vector_1 = Vector([1, 2, 3])
    vector_2 = Vector([4, 5, 6])

    vector_added = vector_1 + vector_2
    vector_multiplied = vector_1 * vector_2
    vector_multiplied_by_scalar = vector_1 * 2
    vector_subtracted = vector_1 - vector_2

    assert vector_added.vector_list == [5, 7, 9]
    assert vector_multiplied == 32
    assert vector_multiplied_by_scalar.vector_list == [2, 4, 6]
    assert vector_subtracted.vector_list == [-3, -3, -3]


    