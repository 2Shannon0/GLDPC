import gldpc_decoder
import numpy as np
from utils import *

def print_matrix(m):
    for i in range(m.shape[0]):
      for j in range(m.shape[1]):
          print("{:>5}".format(round(m[i][j], 1)), end=' ')
      print()

class GLDPC:
    def __init__(self, H_LDPC, H_comp, H_GLDPC=None, CC_DECODER=None):
        self.H_LDPC = H_LDPC

        self.H_comp = H_comp
        self.CC_DECODER = CC_DECODER

        self.H_GLDPC = H_GLDPC if H_GLDPC is not None else self.create_gldpc_matrix(self.H_LDPC, self.H_comp)

        self.row_layer_match = self.create_row_layer_match()

        self.sorted_original_indexes = self.get_sorted_original_indexes()

        # print('\nH_LDPC')
        # print_matrix(self.H_LDPC)

        # print('\nH_GLDPC')
        # print_matrix(self.H_GLDPC)

        # print('\nH_comp')
        # print_matrix(self.H_comp)



    
    def create_gldpc_matrix(self, H_LDPC, H_component):
        m_ldpc, n_ldpc = H_LDPC.shape
        m_comp, n_comp = H_component.shape
        
        H_gldpc = np.zeros((m_ldpc * m_comp, n_ldpc), dtype=int)

        for i in range(m_ldpc):                                                                  
            ones_indices = np.where(H_LDPC[i] == 1)[0]
            num_ones = len(ones_indices)
            selected_columns = np.random.choice(n_comp, size=num_ones, replace=False)
            
            for k, j in enumerate(ones_indices):
                H_gldpc[i * m_comp:(i + 1) * m_comp, j] = H_component[:, selected_columns[k]]

        print('\nH_GLDPC была создана НОВАЯ\n')
        save_to_csv(H_gldpc, '/home/i17m5/GLDPC/matricies/Current_H_gldpc.csv')

        return H_gldpc

    def create_row_layer_match(self):
        row_layer_match = {}
        for i in range(self.H_LDPC.shape[0]):
            layer = self.H_GLDPC[i * self.H_comp.shape[0]:(i + 1) * self.H_comp.shape[0], :]

            column_mapping = {}
            for j, col1 in enumerate(layer.T):
                for k, col2 in enumerate(self.H_comp.T):
                    if np.array_equal(col1, col2):
                        column_mapping[j] = k
                        break

            mapped_pairs = [(k, j) for j, k in column_mapping.items()]
            mapped_pairs.sort()

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

    def get_sorted_original_indexes(self):
        result = []
        for i in range(len(self.row_layer_match)):
            result.append(self.row_layer_match[i]['sorted_original_indexes'])
        return result

    def decode_cpp(self, llr, sigma2, max_iter, use_normalization = True):
        return gldpc_decoder.decode_gldpc(
            self.H_GLDPC,
            self.H_LDPC,
            self.sorted_original_indexes,
            self.CC_DECODER.edg_bpsk,
            llr,
            sigma2,
            max_iter,
            use_normalization
        )
    
    def decode(self, L, sigma2, maxIter):
        m_ldpc, n_ldpc = self.H_LDPC.shape
        H_gamma = np.copy(self.H_LDPC).astype(float)
        H_q = np.copy(self.H_LDPC).astype(float)
        out_L = np.zeros(n_ldpc)

        # Шаг 0 – инициализация значений, пришедших из канала
        for i in range(m_ldpc):
            for j in range(n_ldpc):
                H_q[i,j] = self.H_LDPC[i,j] * L[j]

        # Начало итеративного декодирования
        for iter in range(maxIter):
            # Шаг 1 – передача сообщений от проверочных узлов символьным
            for i in range(m_ldpc):
                sorted_indexes = self.row_layer_match[i]['sorted_original_indexes']
                llr_in_layer_decoder = H_q[i, sorted_indexes]
                # print("llr_in_layer_decoder:", llr_in_layer_decoder)

                llr_from_layer_decoder = self.CC_DECODER.decode(llr_in_layer_decoder, sigma2)
                # print('llr by BCJR decoder:\n', llr_from_layer_decoder)

                H_gamma[i, sorted_indexes] = llr_from_layer_decoder - llr_in_layer_decoder

            # Шаг 2 – передача сообщений от символьных узлов проверочным
            for i in range(m_ldpc):
                for j in range(n_ldpc):
                    if self.H_LDPC[i,j] == 1:
                        indexes = np.setdiff1d(np.nonzero(self.H_LDPC[:,j]), i)
                        H_q[i,j] = L[j] + np.sum(H_gamma[indexes,j])

            # Шаг 3 – формирование синдрома и попытка декодирования
            for i in range(n_ldpc):
                out_L[i] = L[i] +  np.sum(H_gamma[:,i])
            x_hat = np.array(out_L<0, dtype=int)
            print('СИНДРОМ G: ', np.matmul(x_hat, (self.H_GLDPC.T)))
            print('СИНДРОМ G % 2: ', np.matmul(x_hat, (self.H_GLDPC.T)) % 2)
            print('SUM(СИНДРОМ G % 2): ', np.sum(np.matmul(x_hat, (self.H_GLDPC.T)) % 2))

            # print('СИНДРОМ: ', np.matmul(x_hat, (self.H_LDPC.T)))
            # print('СИНДРОМ % 2: ', np.matmul(x_hat, (self.H_LDPC.T)) % 2)
            # print('СИНДРОМ % 2: ', np.matmul(x_hat, (self.H_GLDPC.T)) % 2)


            
            if np.sum(np.matmul(x_hat, (self.H_GLDPC.T))% 2) == 0:
                x = x_hat
                print("\nХОРОШО Ошибки исправлены, кол-во итераций", iter+1)
                print('\nout_L', out_L)
                return x
        print("\nКодовое слово не найдено, кол-во итераций", iter+1)
        print('\nout_L', out_L)
        return np.array(out_L<0, dtype=int)
        

