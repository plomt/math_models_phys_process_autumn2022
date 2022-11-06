"""
Файл для теста функций из main.py
"""
import pytest
import numpy as np

from main import psi, add_degree, matrix_A


@pytest.mark.parametrize("j,x,expected",
                         [
                             [1, 1.0, 0.4338837391175581],
                             [2, 1.0, 0.7818314824680298],
                             [1, 5.0, 0.7818314824680299],
                             [2, 5.0, -0.9749279121818236]
                         ])
def test_psi(j, x, expected):
    result = psi(j, x)
    assert expected == result


@pytest.mark.parametrize("x,n,expected",
                         [
                             [np.array([1, 2]), 1, (2,)],
                             [np.array([1, 2]), 2, (2, 2)],
                             [np.array([1, 2]), 3, (2, 3)]
                         ])
def test_add_agree_shape(x, n, expected):
    result_matrix = add_degree(x, n)
    print(result_matrix)
    assert expected == result_matrix.shape


@pytest.mark.parametrize("n,x,C,expected",
                         [
                             [2, np.array([1, 2, 3]), np.array([1, 2, 3]), 2],
                             [3, np.array([1, 2, 3]), np.array([1, 2, 3]), 3],
                             [4, np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4]), 4]
                         ])
def test_matrix_A_shape(n, x, C, expected):
    result = matrix_A(n, x, C)
    assert expected == result.shape[0]


def test_fi():
    pass
