import numpy as np
import scipy as sp
import scipy.sparse as sparse
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import erf


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

            b[r_ind] -= (ij_coef - 2 * h ** 2 / dt) * get_C_ij_prev(i, j)  # C^{n}_ij
            b[r_ind] -= ipj_coef * get_C_ij_prev(i - 1, j)  # C^{n}_i-1j
            b[r_ind] -= inj_coef * get_C_ij_prev(i + 1, j)  # C^{n}_i+1j
            b[r_ind] -= y_coef * get_C_ij_prev(i, j + 1)  # C^{n}_ij-1
            b[r_ind] -= y_coef * get_C_ij_prev(i, j - 1)  # C^{n}_ij+1

    row = np.array(row)
    col = np.array(col)
    values = np.array(values)

    return sparse.coo_matrix((values, (row, col)), shape=(N ** 2, N ** 2)), b


def MFV_with_time(reg=False):
    global C
    for tn in range(N_t):
        print(tn)
        A, b = create_MFV_matrix(reg=reg)
        A = A.tocsr()

        # ILU = sparse.linalg.spilu(A, fill_factor=1.7, drop_tol=1e-3)
        # prec = sparse.linalg.LinearOperator((N * N, N * N), matvec=ILU.solve)

        # C, info = sparse.linalg.bicg(A, b, tol=1e-6, M=prec)
        C = sparse.linalg.spsolve(A, b)
    return C


def plot_sol(C_to_plot, name='MFV'):
    x_d = 80
    y_d = 30

    sol = []
    for j in range(N):
        for i in range(N):
            x = x_min + (i + 1) * h
            y = y_min + (j + 1) * h
            if abs(y) <= y_d and x < x_d:
                if i == 0:
                    sol.append([])
                sol[-1].append(C_to_plot[get_ind_by_mind(i, j)])
    sns.heatmap(sol, square=True, cmap="viridis")
    plt.savefig(f"{name}, reg={reg}, N={N}, T={t}, N_t={N_t}, dx={dx}, dy={dy}.png")
    # plt.show()


def calc_ans_in_point(x, y, M=1000):
    def under_int(tao):
        nonlocal x, y
        return tao ** (-1.5) * (erf((a + y) / np.sqrt(4 * dy * tao)) + erf((a - y) / np.sqrt(4 * dy * tao))) \
               * np.exp(-((x - tao) ** 2 / (4 * dx * tao)))

    n_j_array = [np.cos((2 * j - 1) * np.pi / 2 / M) for j in range(1, M + 1)]
    return x * c_0 / np.sqrt(16 * np.pi * dx) * np.pi * t / 2 / M \
            * np.sum(np.array([np.sqrt(1-np.power(n_j, 2)) * under_int(t * (n_j + 1) / 2)
                               for n_j in n_j_array]))


def calc_ans():
    x_d = 80
    y_d = 30

    C_form = np.zeros(N * N, dtype=float)
    for j in range(N):
        print(j)
        for i in range(N):
            x = x_min + (i + 1) * h
            y = y_min + (j + 1) * h
            C_form[get_ind_by_mind(i, j)] = calc_ans_in_point(x, y)
    return C_form



x_min = 0
x_max = 200
y_min = -100
y_max = 100

N = 200
h = (x_max - x_min) / N

dx = 1
dy = 0.1

t = 50
N_t = 50
dt = t / N_t

C = np.zeros(N * N, dtype=float)
c_0 = 1
a = 10
reg = True

"""
C_ans = MFV_with_time(reg)
np.save('MFV_1', C_ans)
plot_sol(C_ans)
"""
"""
C_form = calc_ans()
np.save('ans_1', C_form)
plot_sol(C_form, name='ANS')
"""
C_ans = np.load('ans_1.npy')
C_mfv = np.load('MFV_1.npy')
plot_sol(np.abs(C_mfv - C_ans), name='DIFF')
print(np.max(np.abs(C_mfv - C_ans)))
