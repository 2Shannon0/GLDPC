import numpy as np
import csv
from utils import *
def print_matrix(m):
    for i in range(m.shape[0]):
      for j in range(m.shape[1]):
          print("{:>5}".format(round(m[i][j], 1)), end=' ')
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

class GLDPC:
    def __init__(self, H_LDPC, H_comp, H_GLDPC=None, CC_DECODER=None):
        self.H_LDPC = H_LDPC

        self.H_comp = H_comp
        self.CC_DECODER = CC_DECODER

        self.H_GLDPC = H_GLDPC if H_GLDPC is not None else self.create_gldpc_matrix(self.H_LDPC, self.H_comp)

        self.row_layer_match = self.create_row_layer_match()

        print('\nH_LDPC')
        print_matrix(self.H_LDPC)

        print('\nH_GLDPC')
        print_matrix(self.H_GLDPC)

        print('\nH_comp')
        print_matrix(self.H_comp)



    
    def create_gldpc_matrix(self, H_LDPC, H_component):
        # Размеры матриц
        m_ldpc, n_ldpc = H_LDPC.shape  # m_ldpc строк, n_ldpc столбцов
        m_comp, n_comp = H_component.shape  # m_comp строк, n_comp столбцов
        
        # Итоговая матрица GLDPC: m_ldpc * m_comp строк, n_ldpc столбцов
        H_gldpc = np.zeros((m_ldpc * m_comp, n_ldpc), dtype=int)
        # print("Соответствие строк H_LDPC к слоям H_component:\n")
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
        print('\nH_GLDPC была создана новая\n')
        return H_gldpc
    
    # def create_row_layer_match(self):
    #     row_layer_match = {}
    #     print('\n---row_layer_match---\n')
        
    #     for i in range(self.H_LDPC.shape[0]):
    #         layer = self.H_GLDPC[i * self.H_comp.shape[0]:(i + 1) * self.H_comp.shape[0], :]

    #         column_mapping = {}
    #         for j, col1 in enumerate(layer.T):
    #             for k, col2 in enumerate(self.H_comp.T):
    #                 if np.array_equal(col1, col2):
    #                     column_mapping[j] = k
    #                     break
            
    #         row_layer_match[i] = {"layer": layer, "column_mapping": column_mapping}
            
    #         print(f'строке H_GLDPC №{i} соответствует слой')
    #         print_matrix(layer)
    #         print(column_mapping)
    def create_row_layer_match(self):
        row_layer_match = {}
        # print('\n---row_layer_match---\n')
        for i in range(self.H_LDPC.shape[0]):
            layer = self.H_GLDPC[i * self.H_comp.shape[0]:(i + 1) * self.H_comp.shape[0], :]

            column_mapping = {}
            for j, col1 in enumerate(layer.T):
                for k, col2 in enumerate(self.H_comp.T):
                    if np.array_equal(col1, col2):
                        column_mapping[j] = k
                        break



            # Создаем пары (индекс в H_comp, индекс в layer), чтобы сортировать по H_comp
            mapped_pairs = [(k, j) for j, k in column_mapping.items()]
            mapped_pairs.sort()  # Упорядочиваем по индексу в H_comp

            # Разбиваем на два массива (H_comp -> layer)
            # sorted_original_indexes - массив индексов слоя в порядке как в H_comp
            sorted_mapped_indexes, sorted_original_indexes = zip(*mapped_pairs) if mapped_pairs else ([], [])

            row_layer_match[i] = {
                "layer": layer,
                "column_mapping": column_mapping,
                "sorted_original_indexes": np.array(sorted_original_indexes),  # Индексы в layer
                "sorted_mapped_indexes": np.array(sorted_mapped_indexes),      # Индексы в H_comp
            }

            # print(f'Строке H_GLDPC №{i} соответствует слой:')
            # print_matrix(layer)
            # print('column_mapping:', column_mapping)
            # print('sorted_mapped_indexes:', sorted_mapped_indexes)
            # print('sorted_original_indexes:', sorted_original_indexes)

        return row_layer_match

    
    def decode(self, L, sigma2, maxIter):
        m_ldpc, n_ldpc = self.H_LDPC.shape
        H_gamma = np.copy(self.H_LDPC).astype(float)
        # print('\nH_gamma изначально:')
        # print_matrix(H_gamma)

        H_q = np.copy(self.H_LDPC).astype(float)
        # print('\nH_q изначально:')
        # print_matrix(H_q)
        # print('\nllr_in: ', L)

        
        out_L = np.zeros(n_ldpc)
        # print('out_L')
        # print(out_L)


        for i in range(m_ldpc):
            for j in range(n_ldpc):
                H_q[i,j] = self.H_LDPC[i,j] * L[j]
        # print('\nH_q после инициализации:')
        # print_matrix(H_q)

        for iter in range(maxIter):
            # for i in range(m_ldpc):
            #     llr_in_layer_decoder = H_q[i, self.row_layer_match[i]['sorted_original_indexes']]
            #     print("llr_in_layer_decoder:", llr_in_layer_decoder)
            #     llr_from_layer_decoder = self.CC_DECODER.decode(llr_in_layer_decoder, sigma2)

            #     print('llr by bcjr decoder:\n', llr_from_layer_decoder)
            #     for j in range(len(llr_from_layer_decoder)):
            #         H_gamma[i, self.row_layer_match[i]['sorted_original_indexes'][j]] = llr_from_layer_decoder[j]
            for i in range(m_ldpc):
                sorted_indexes = self.row_layer_match[i]['sorted_original_indexes']

                # Берем соответствующие значения из H_q
                llr_in_layer_decoder = H_q[i, sorted_indexes]
                # print("llr_in_layer_decoder:", llr_in_layer_decoder)

                # Декодируем с помощью BCJR
                llr_from_layer_decoder = self.CC_DECODER.decode(llr_in_layer_decoder, sigma2)
                # print('llr by BCJR decoder:\n', llr_from_layer_decoder)

                # Записываем обратно в H_gamma по **изначальному** индексу
                # H_gamma[i, sorted_indexes] = llr_from_layer_decoder
                H_gamma[i, sorted_indexes] = llr_from_layer_decoder - llr_in_layer_decoder # сделаем как бы экстринсик LLR

      
            # print('H_gamma')
            # print_matrix(H_gamma)

            for i in range(m_ldpc):
                for j in range(n_ldpc):
                    if self.H_LDPC[i,j] == 1:
                        indexes = np.setdiff1d(np.nonzero(self.H_LDPC[:,j]), i)
                        H_q[i,j] = L[j] + 0.1*np.sum(H_gamma[indexes,j])
            # print('H_q')
            # print_matrix(H_q)

            for i in range(n_ldpc):
                out_L[i] = L[i] +  np.sum(H_gamma[:,i])

            x_hat = np.array(out_L<0, dtype=int)

            #ВЫВОДИТЬ КОДОВОЕ СЛОВО НА КАЖДОМ ШАГЕ
            # print('\n',x_hat)
            if np.sum(np.matmul(x_hat, (self.H_LDPC.T)) % 2) == 0:
                x = x_hat
                # print("\nОшибки исправлены, кол-во итераций", iter+1)
                # print('\nout_L', out_L)
                return x
        # print("\nКодовое слово не найдено, кол-во итераций", iter+1)
        # print('\nout_L', out_L)
