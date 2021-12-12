import numpy as np
import scipy as sp
import scipy.sparse as sparse
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.integrate import nquad


def test_inside(i, j):
    if 0 <= i < N and 0 <= j < N:
        return True
    return False


def test_in(i, j):
    if 0 <= i <= N and 0 <= j <= N:
        return True
    return False


def test_in_bound_D(i, j):
    if i == N or j == N:
        return True
    return False


def get_multi_index(i, j):
    return i + N * j


def get_indexes_by_multi_index(index):
    return index % N, index // N


def create_matrix_FEM():
    """
    Функция создания матрицы метода моментов. Работает без вычисления интегралов (посчитаны руками), матрица
    симметричная, положительно определённая и совпадает с базовым вариантом через численное интегрирование, но она
    не обновлена под вычисление интегралов только внутри [0, 1] x [0, 1]
    :return: A - матрица метода моментов, COO-format
    """
    row = []
    col = []
    values = []
    b = np.zeros(shape=N ** 2)
    integral_dict = {(0, 0): 44 / 3, (0, 1): -19 / 3, (1, 1): -55 / 12, (1, 0): 8 / 3, (1, -1): 11 / 12}

    def append_COO(i, j, k, l):
        if test_in(i, j) and test_in(k, l):
            delta_x = k - i
            delta_y = l - j
            if (delta_x, delta_y) in integral_dict:
                val = integral_dict[(delta_x, delta_y)]
            else:
                val = integral_dict[(-delta_x, -delta_y)]

            if test_inside(k, l):
                row.append(get_multi_index(i, j))
                col.append(get_multi_index(k, l))
                values.append(val)
            else:
                x_k, y_l = coords_by_inside(k, l)
                b[get_multi_index(i, j)] -= val * answer(x_k, y_l)

    for i in range(0, N):
        for j in range(0, N):
            for k in range(i - 1, i + 2):
                for l in range(j - 1, j + 2):
                    append_COO(i, j, k, l)

    row = np.array(row)
    col = np.array(col)
    values = np.array(values)
    # values *= N ** 2
    return sparse.coo_matrix((values, (row, col)), shape=(N ** 2, N ** 2)), b


def plot_matrix(matrix):
    dens_matrix = matrix.toarray()
    sns.heatmap(dens_matrix != 0, cmap="Greys", square=True, annot=False)
    plt.show()


def coords_by_inside(i, j):
    return i * h, j * h


def basis_function(i, j, x, y):
    x_i, y_j = coords_by_inside(i, j)
    x_r = (x - x_i) / h
    y_r = (y - y_j) / h
    if x > 1 or x < 0 or y < 0 or y > 1:
        return 0
    if abs(x_r) > 1 or abs(y_r) > 1:  # Не supp \phi
        return 0
    if x_r >= 0 and y_r >= 0:  # 0
        return (1 - x_r) * (1 - y_r)
    elif x_r >= 0 and y_r < 0:  # 1
        return (1 - x_r) * (1 + y_r)
    elif x_r < 0 and y_r < 0:  # 2
        return (1 + x_r) * (1 + y_r)
    else:  # 3
        return (1 + x_r) * (1 - y_r)


def div_basis_function(i, j, x, y, tensor=False):
    """
    Вычисление (D grad, phi_ij)(x, y). Параметр tensor отвечает за применение тензора диффуции.
    """
    x_i, y_j = coords_by_inside(i, j)
    x_r = (x - x_i) / h
    y_r = (y - y_j) / h
    if x > 1 or x < 0 or y < 0 or y > 1:  # Вот эта строка отвечает, за отдельную обработку границы
        return 0                          # Если её убрать, то с матрицей всё в порядке, но интеграл берётся по
                                          # по пересечению носителей базовых функций за границе [0, 1] x [0, 1]
    if abs(x_r) > 1 or abs(y_r) > 1:  # Не supp \phi
        return 0
    if x_r >= 0 and y_r >= 0:  # 0
        d_dx = -(1 - y_r)
        d_dy = -(1 - x_r)
    elif x_r >= 0 and y_r < 0:  # 1
        d_dx = -(1 + y_r)
        d_dy = (1 - x_r)
    elif x_r < 0 and y_r < 0:  # 2
        d_dx = (1 + y_r)
        d_dy = (1 + x_r)
    else:  # 3
        d_dx = (1 - y_r)
        d_dy = -(1 + x_r)
    if tensor:
        return 1 / h * (d_dx + 10 * d_dy)
    return 1 / h * (d_dx + d_dy)


def create_matrix_FEM_with_quad():
    """
    Базовая версия создания матрицы метода мом  ентов. Работает с помощью численного интегрирования, максимально медленно
    и не оптимально, но создана только для проверки оптимизированной функции. Учитывает, что supp phi_ij принаждлежит
    [0, 1] x [0, 1]. Матрица получается не симметричной.
    :return: A - матрица метода моментов, COO-format
    """
    b = np.zeros(N ** 2)
    row = []
    col = []
    values = []

    def diff_operator(x, y):
        nonlocal i, j, k, l
        return div_basis_function(k, l, x, y, tensor=True) * div_basis_function(i, j, x, y, tensor=False)

    for i in range(0, N):
        for j in range(0, N):
            for k in range(i - 1, i + 2):
                for l in range(j - 1, j + 2):
                    x_i, y_j = coords_by_inside(i, j)
                    x_k, y_l = coords_by_inside(k, l)
                    bounds_x = [max(x_i - h, x_k - h, 0), min(x_i + h, x_k + h, 1)]
                    bounds_y = [max(y_j - h, y_l - h, 0), min(y_j + h, y_l + h, 1)]
                    I_by_quad, error = nquad(diff_operator, [bounds_x, bounds_y])
                    if test_inside(k, l):
                        row.append(get_multi_index(i, j))
                        col.append(get_multi_index(k, l))
                        values.append(I_by_quad)
                    elif test_in(k, l) and test_in_bound_D(k, l):
                        index = get_multi_index(i, j)
                        b[index] -= I_by_quad * answer(x_k, y_l)

    row = np.array(row)
    col = np.array(col)
    values = np.array(values)
    return sparse.coo_matrix((values, (row, col)), shape=(N ** 2, N ** 2)), b


def answer(x, y):
    return np.cos(np.pi * x) * np.cos(np.pi * y)


def f(x, y):
    return 11 * np.pi ** 2 * answer(x, y)


def create_right(b):

    def f_scalar_basic_ij(x, y):
        nonlocal i, j
        return f(x, y) * basis_function(i, j, x, y)

    for i in range(0, N):
        for j in range(0, N):
            index = get_multi_index(i, j)
            # (f, \phi_ij)
            x_i, y_j = coords_by_inside(i, j)
            bounds_x = [x_i - h, x_i + h]
            bounds_y = [y_j - h, y_j + h]
            I_by_quad, error = nquad(f_scalar_basic_ij, [bounds_x, bounds_y])
            b[index] += I_by_quad

            # Нейман
            # g_N = 0
    return b


def calc_c_norm(x):
    error = 0
    for i in range(0, N):
        for j in range(0, N):
            index = get_multi_index(i, j)
            x_i, y_j = coords_by_inside(i, j)
            error = max(error, np.abs(x[index] - answer(x_i, y_j)))
    return error


N = 64
h = 1 / N

A_quad, b_init = create_matrix_FEM_with_quad()
A_quad = A_quad.tocsr()
print(b_init)
# print(np.linalg.norm(A.toarray() - A_quad.toarray()))
# sns.heatmap(np.abs(A.toarray() - A_quad.toarray()), square=True)
# plt.show()
# print(A.toarray())
print(A_quad.toarray()[:, 0])


b_quad = create_right(b_init)
x_real = np.zeros(shape=(N ** 2))
for i in range(0, N):
    for j in range(0, N):
        index = get_multi_index(i, j)
        x_i, y_j = coords_by_inside(i, j)
        x_real[index] = answer(x_i, y_j)
b_for_real = A_quad @ x_real

print("Ошибка на правой части:")
print(np.linalg.norm(b_quad - b_for_real))
# print(np.linalg.norm(b - b_for_real))
print(b_quad)
# print(np.round(b_for_real, 2))
print(b_for_real)

#ILU = sparse.linalg.spilu(A_quad, fill_factor=1.7, drop_tol=1e-3)
#prec = sparse.linalg.LinearOperator((N * N, N * N), matvec=ILU.solve)

#x, info = sparse.linalg.bicg(A_quad, b_old, tol=1e-6, M=prec)
x = sparse.linalg.spsolve(A_quad, b_quad)
print("Невязка: ")
print(np.linalg.norm(A_quad @ x - b_quad))
# print("info: ", info)
# print(calc_c_norm(x))
print("Норма ошибки (чебышёвская): ")
print(calc_c_norm(x))

error_matrix = np.zeros(shape=(N, N))
for i in range(0, N):
    for j in range(0, N):
        index = get_multi_index(i, j)
        x_i, y_j = coords_by_inside(i, j)
        error_matrix[i, j] = np.abs(x[index] - answer(x_i, y_j))

sns.heatmap(error_matrix, cmap="Greys", square=True)
plt.show()
