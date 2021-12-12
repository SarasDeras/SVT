import numpy as np
import scipy as sp
import scipy.sparse as sparse
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.integrate import nquad


def get_ind_by_mind(i, j):
    return i + N * j


def get_mind_by_ind(index):
    return index % N, index // N


def get_C_ij_prev(i, j) -> float:
    if 0 <= i < N and 0 <= j < N:
        return C[get_ind_by_mind(i, j)]
    elif i == -1 and 0 <= j < N:
        if abs((j + 1 / 2) * h + y_min) < a:
            return c_0
        else:
            return 0
    elif (i == N and 0 <= j < N) or ((j == N or j == -1) and 0 <= i < N):
        return 0
    else:
        raise IndexError


def create_MFV_matrix(reg=False):
    row = []
    col = []
    values = []
    b = np.zeros(N ** 2, dtype=float)

    def append_COO(r_ind, i, j, val):
        row.append(r_ind)
        col.append(get_ind_by_mind(i, j))
        values.append(val)

    for i in range(N):
        for j in range(N):
            r_ind = get_ind_by_mind(i, j)
            ij_coef = h ** 2 / dt + dx + dy
            ipj_coef = -h / 4 - dx / 2
            inj_coef = h / 4 - dx / 2
            y_coef = -dy / 2

            if reg:
                ij_coef = h ** 2 / dt + dx + dy + h / 2
                inj_coef = -dx / 2
                ipj_coef = -h / 2 - dx / 2

            append_COO(r_ind, i, j, ij_coef)  # C^{n + 1}_ij

            if i != 0:
                append_COO(r_ind, i - 1, j, ipj_coef)  # C^{n + 1}_i-1j
            else:
                b[r_ind] -= get_C_ij_prev(i - 1, j) * ipj_coef

            if i != N - 1:
                append_COO(r_ind, i + 1, j, inj_coef)  # C^{n + 1}_i+1j

            if j != N - 1:
                append_COO(r_ind, i, j + 1, y_coef)  # C^{n + 1}_ij+1

            if j != 0:
                append_COO(r_ind, i, j - 1, y_coef)  # C^{n + 1}_ij-1

            b[r_ind] -= (inj_coef - h ** 2) * get_C_ij_prev(i, j)  # C^{n}_ij
            b[r_ind] -= ipj_coef * get_C_ij_prev(i - 1, j)  # C^{n}_i-1j
            b[r_ind] -= inj_coef * get_C_ij_prev(i + 1, j)  # C^{n}_i+1j
            b[r_ind] -= y_coef * get_C_ij_prev(i, j + 1)  # C^{n}_ij-1
            b[r_ind] -= y_coef * get_C_ij_prev(i, j - 1)  # C^{n}_ij+1

    row = np.array(row)
    col = np.array(col)
    values = np.array(values)

    return sparse.coo_matrix((values, (row, col)), shape=(N ** 2, N ** 2)), b


def MFV_with_time():
    global C
    for tn in range(N_t):
        print(tn)
        A, b = create_MFV_matrix()
        A = A.tocsr()

        # ILU = sparse.linalg.spilu(A_quad, fill_factor=1.7, drop_tol=1e-3)
        # prec = sparse.linalg.LinearOperator((N * N, N * N), matvec=ILU.solve)

        # x, info = sparse.linalg.bicg(A_quad, b_old, tol=1e-6, M=prec)
        C = sparse.linalg.spsolve(A, b)
    return C


def plot_sol():
    x_d = 80
    y_d = 30

    sol = []
    for j in range(N):
        for i in range(N):
            x = x_min + i * h
            y = y_min + j * h
            if abs(y) <= y_d and x < x_d:
                if i == 0:
                    sol.append([])
                sol[-1].append(C[get_ind_by_mind(i, j)])
    sns.heatmap(sol, square=True, cmap="viridis")
    plt.show()


x_min = 0
x_max = 200
y_min = -100
y_max = 100

N = 200
h = (x_max - x_min) / N

dx = 1e-4
dy = 0.1

t = 50
N_t = 100
dt = t / N_t

C = np.zeros(N * N, dtype=float)
c_0 = 1
a = 10

C_ans = MFV_with_time()
plot_sol()
