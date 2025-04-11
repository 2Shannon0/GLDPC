import numpy as np
from utils import read_csv

# Загрузка матрицы
h_gldpc = read_csv('/home/i17m5/GLDPC/matricies/H_gldpc_like_example.csv')
H = np.array(h_gldpc).reshape((20, 32)) % 2  # Убеждаемся, что в GF(2)

# Функция для вычисления базиса ядра в GF(2)
def gf2_nullspace(H):
    # Приведение к ступенчатому виду в GF(2)
    m, n = H.shape
    H = H.copy()
    pivots = []
    for i in range(m):
        pivot = -1
        for j in range(n):
            if H[i, j] == 1:
                pivot = j
                break
        if pivot == -1:
            continue
        pivots.append(pivot)
        for k in range(m):
            if k != i and H[k, pivot] == 1:
                H[k, :] = (H[k, :] + H[i, :]) % 2

    rank = len(pivots)
    free_vars = [j for j in range(n) if j not in pivots]
    nullity = len(free_vars)
    basis = []

    for free_idx in free_vars:
        x = np.zeros(n, dtype=int)
        x[free_idx] = 1
        for pivot_idx, row in zip(pivots, range(rank)):
            if H[row, free_idx] == 1:
                x[pivot_idx] = 1
        basis.append(x)

    return np.array(basis).T

# Вычисление базиса
nullspace = gf2_nullspace(H)
print(f"Размер матрицы H_GLDPC: {H.shape}")
print(f"Ранг матрицы: {H.shape[0] - nullspace.shape[1]}")
print(f"Размерность ядра: {nullspace.shape[1]}")

# Вывод базиса и проверка
print("\nБазис нулевого пространства:")
for i in range(nullspace.shape[1]):
    x = nullspace[:, i]
    check = np.matmul(H, x) % 2
    print(f"\nВектор {i + 1}:")
    print(x)
    print(f"Проверка H_GLDPC * x = 0: {check}")
    print(f"Сумма проверки: {np.sum(check)}")

# Пример ненулевого кодового слова
nonzero_words = [x for x in nullspace.T if np.sum(x) > 0]
if nonzero_words:
    print("\nПример ненулевого кодового слова:")
    print(nonzero_words[0])