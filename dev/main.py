"""
Восстановление утраченного показания высотного датчика контроля за полем нейтронов по данным архива.
Реактор: ВВЭР
"""
import math

import numpy as np
import pandas as pd

H = 7  # высота реактора


def psi(j: int, x: float) -> float:
    """
    Собственные функции реактора
    :param j: номер итерации(датчика)
    :type j: int
    :param x: аргумент функции
    :type x: float
    :return: значение собственной функции
    :rtype: float
    """
    return np.sin(j * np.pi * x / H)


def add_degree(x: np.ndarray, n: int):
    """
    Возвращает матрицу признаков для расчета коэффициентов аппроксимации
    :param x:
    :type x: float
    :param n: число собственных функций
    :type n: int
    :return: матрица признаков
    :rtype: np.array
    """
    seq = map(lambda y: np.sin(1 * np.pi * y / H), x)
    res_X = np.fromiter(seq, dtype=float)
    for iteration in range(2, n + 1):
        seq = map(lambda y: np.sin(iteration * np.pi * y / H), x)
        add_col = np.fromiter(seq, dtype=float)
        res_X = np.column_stack((res_X, add_col))
    return res_X


def matrix_A(n: int, x: np.ndarray, C: np.array) -> np.array:
    """
    Нахождение коэффициентов аппроксимации по методу наименьших квадратов
    :param n: число собственных функций
    :type n: int
    :param x: значение высоты датчика. [0, H]
    :type x: float
    :param C: показания работоспособных датчиков
    :type C: np.array
    :return: коэффициенты аппроксимации
    :rtype: np.array
    """
    assert n <= C.shape[0], "Число функций разложения должно быть не больше числа работающих датчиков"
    X: np.array = add_degree(x, n)
    return np.linalg.inv(np.transpose(X) @ X) @ np.transpose(X) @ C


def fi(x: float, n: int) -> float:
    """
    Плотность потока нейтронов по высоте.
    Аппроксимация с помощью собственных функций одномерного
    реактора с однородной загрузкой без обратных связей.
    :param x: аргумент функции
    :type x: float
    :param n: число собственных функций
    :type n: int
    :return: значение распределения в точке x
    :rtype: float
    """
    # TODO: вместо 9999 подставить значение коэффициента аппроксимации
    return sum([9999 * psi(j, x) for j in range(1, n + 1)])


def get_corr_matrix(matrix_A: np.array) -> np.array:
    """
    Расчет матрицы коэффициентов корреляции амплитуд
    :param matrix_A:
    :type matrix_A:
    :return:
    :rtype:
    """
    return np.corrcoef(matrix_A)

print(matrix_A(4, np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4])))