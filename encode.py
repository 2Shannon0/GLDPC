import numpy as np
from utils import read_csv, print_matrix, save_to_csv

def create_random_code_word(k):
    return np.round(np.random.rand(k))

def print_matrix_with_commas(m):
    print()
    for i in range(m.shape[0]):
      print('[', end = '')
      for j in range(m.shape[1]):
          if j != m.shape[1] - 1:
              print(int(m[i][j]), end=", ")
          else:
              print(int(m[i][j]), end="")
      if i != m.shape[0] - 1:
          print('], ')  
      else:
          print("]")  
      print() 

def print_array_with_commas(a):
    print('[', end = '' )
    for i in range(a.shape[0]):
        if i != a.shape[0] - 1:
            print(int(a[i]),end=", ")  
        else:
            print(int(a[i]), end="]")
    print()

H = read_csv('/home/i17m5/GLDPC/matricies/H_GLDPC_from_LDPC(420,364)_BCH(15,11).csv')
# H = read_csv('/home/i17m5/GLDPC/matricies/H_ham(16,11).csv')

# H = np.array(h_gldpc) 

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
for i in range(nullspace.shape[1]):
    if np.sum(np.matmul(nullspace[:, i], (H.T)) % 2) == 0:
        continue
        # print(nullspace[:, i])
    else: 
        raise(Exception)

print('Базис:\n')
print(f"Размер базис: {nullspace.shape}")
print_matrix(nullspace)
# print_matrix_with_commas(nullspace.T)
print(f"Размер матрицы H: {H.shape}")
print(f"Ранг матрицы: {H.shape[0] - nullspace.shape[1]}")

G = nullspace.T
save_to_csv(G, '/home/i17m5/GLDPC/matricies/Current_G_gldpc.csv')
print_matrix(G)
# print_matrix_with_commas(G)

# info = [1,1,1,1,1,1,1,1,1,1,1,1,0]
info = create_random_code_word(G.shape[0])
print('Информационное слово:')
print_array_with_commas(info)
codeword = np.matmul(info, G) % 2

if np.sum(np.matmul(codeword, (H.T)) % 2) == 0:
    print('Верное кодовое слово:')
    print_array_with_commas(codeword)
else:
    print('Кодовое слово с ошибкой\n', np.matmul(codeword, (H.T)) % 2)

