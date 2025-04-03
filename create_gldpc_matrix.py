from utils import *

def save_to_csv(matrix, filepath):
    np.savetxt(filepath, matrix, delimiter=',', fmt='%d')

def print_matrix(m):
    for i in range(m.shape[0]):
      for j in range(m.shape[1]):
          print("{:>3}".format(round(m[i][j], 5)), end=' ')
      print()

def read_csv( filename ):
    with open( filename, newline='') as f:
        reader = csv.reader(f)
        symbols = []
        for row in reader:
            rows = []
            for symbol in row:
                rows.append(int(symbol))
            symbols.append(rows)
    return np.array(symbols)

h_ldpc = read_csv('/home/i17m5/GLDPC/matricies/H_LDPC(32,28).csv')
h_component = read_csv('/home/i17m5/GLDPC/matricies/H_ham(16,11).csv')

# print_matrix(h_component)
print('\n\n\n\n')
# print_matrix(h_ldpc)


def create_gldpc_matrix(H_LDPC, H_component):
    # Размеры матриц
    m_ldpc, n_ldpc = H_LDPC.shape  # m_ldpc строк, n_ldpc столбцов
    m_comp, n_comp = H_component.shape  # m_comp строк, n_comp столбцов
    
    # Итоговая матрица GLDPC: m_ldpc * m_comp строк, n_ldpc столбцов
    H_gldpc = np.zeros((m_ldpc * m_comp, n_ldpc), dtype=int)
    
    # Для каждой строки H_LDPC
    for i in range(m_ldpc):
        # Находим индексы единиц в строке i
        ones_indices = np.where(H_LDPC[i] == 1)[0]
        # print('ones_indices on iter: ', i, '\n', ones_indices)
        num_ones = len(ones_indices)
        
        # Проверяем, что количество единиц не превышает число столбцов в H_component
        if num_ones > n_comp:
            raise ValueError(f"В строке {i} H_LDPC слишком много единиц -({num_ones}) для замены столбцами H_component, которых ({n_comp})")
        
        # Случайно выбираем уникальные столбцы из H_component
        selected_columns = np.random.choice(n_comp, size=num_ones, replace=False)
        # print('selected_columns on iter: ', i, '\n', selected_columns)

        
        # Заполняем блок H_gldpc для текущей строки H_LDPC
        for k, j in enumerate(ones_indices):
            # Блок строк от i*m_comp до (i+1)*m_comp, столбец j
            # print(H_gldpc[i * m_comp:(i + 1) * m_comp, j])

            H_gldpc[i * m_comp:(i + 1) * m_comp, j] = H_component[:, selected_columns[k]]
    
    return H_gldpc

# Создание матрицы GLDPC
H_gldpc = create_gldpc_matrix(h_ldpc, h_component)
print('\n\n\nМатрица H_GLPDC:')
print_matrix(H_gldpc)

# save_to_csv(H_gldpc, '/home/i17m5/GLDPC/matricies/H_gldpc.csv')