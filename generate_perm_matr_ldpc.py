import numpy as np
from utils import read_csv, print_matrix, save_to_csv
def print_matrix(m):
    for i in range(m.shape[0]):
      for j in range(m.shape[1]):
          print("{:>3}".format(round(m[i][j], 1)), end=' ')
      print()

m_1 = 2 # число единиц в столбце
n_1 = 15 # число единиц в строке
n_perm = 28 # размер матрицы перестановок
# original_values = np.random.randint(0, n_perm, size=(m_1, n_1))
# save_to_csv(original_values, '/home/i17m5/GLDPC/matricies/cur_perms_values.csv')
original_values = read_csv('/home/i17m5/GLDPC/matricies/random_perms_values_1.csv')


# identity = np.eye(n_perm, dtype=int)
# print('\nЕдиничная:\n')
# print_matrix(identity)

# shifted = np.roll(identity, 0, axis=1)
# print('\nСо сдвигом:\n')
# print_matrix(shifted)

# Шаг 2: создаём итоговую матрицу 56x42 (2*28 на 15*28)
final_matrix = np.zeros((m_1 * n_perm, n_1 * n_perm), dtype=int)
# print("Итоговая матрица размера:", final_matrix.shape)
# save_to_csv(final_matrix, '/home/i17m5/GLDPC/matricies/z1.csv')

# Шаг 3: заполняем финальную матрицу сдвинутыми единичными матрицами
for i in range(m_1):
    for j in range(n_1):
        shift = original_values[i][j]
        print(shift)
        # Единичная матрица
        identity = np.eye(n_perm, dtype=int)

        # Циклический сдвиг по строкам
        shifted = np.roll(identity, shift, axis=1)

        # Вычисляем, куда вставлять в итоговую матрицу
        row_start = i * n_perm
        col_start = j * n_perm

        final_matrix[row_start:row_start + n_perm, col_start:col_start + n_perm] = shifted

save_to_csv(final_matrix, '/home/i17m5/GLDPC/matricies/cur_perm_matr_ldpc.csv')
print("Фрагмент итоговой матрицы (верхний левый блок 28x28):")
print_matrix(final_matrix[28:56, :n_perm])
