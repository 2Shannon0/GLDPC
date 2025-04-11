from belief_propagation import BP
from utils import *

H = read_csv('/home/i17m5/GLDPC/matricies/BCH_MATRIX_N_31_K_16_DEFAULT.csv')

N = H.shape[1]
print('H\n')
print_matrix(H)

codeword_initial = np.array([0] * N)
print(f'Кодовое слово: {codeword_initial}\n\n')

# codeword_modulated = bpsk_modulation(codeword_initial)
# print(f'Кодовое слово BPSK: {codeword_modulated}\n\n')

llr_in, error_indecies = awgn(codeword_initial, 0.63)
print(f"Количество ошибок {len(error_indecies)}. Позиции ошибок {error_indecies}")
for error in error_indecies:
    print(llr_in[error], end=", ")

print(f'\nВсе входные LLR: {llr_in}\n\n')

sp = BP(llr_in, H, 30)
decoded_word = sp.decode()
print("Декодированное слово")
print(decoded_word)
print("Изначальное кодовое слово")
print(np.array(codeword_initial))
print("Получено правильное кодовое слово ", np.array_equal(decoded_word, codeword_initial))

