from bpsk import bpsk_modulation, bpsk_demodulation
from awgn import awgn_llr
from belief_propagation import BP
from utils import *
from trellis_repo import get_trellis
from BCJR import BCJRDecoder

# H = read_csv('/home/i17m5/GLDPC/matricies/H_GLDPC.csv')
H = read_csv('/home/i17m5/GLDPC/matricies/H_LDPC(32,28).csv')


N = H.shape[1]
print('H\n')
print_matrix(H)

# Создаем декодер кода компонента
# Подгуржаем решетку
# trellis1 = get_trellis('/home/i17m5/GLDPC/trellis_binaries/H_ham(16,11)')
# code_component_decoder = BCJRDecoder(trellis1.edg)

codeword_initial = np.array([0] * N, dtype=int)
# codeword_initial = np.array([1,0,1,0,1,1,1,1])

print(f'Кодовое слово: {codeword_initial}\n\n')

codeword_modulated = bpsk_modulation(codeword_initial)
print(f'Кодовое слово BPSK: {codeword_modulated}\n\n')

#------------------------------------Cоздание входных LLR----------------------------------------------------------------
snr = 2
llr_in, sigma2 = awgn_llr(codeword_modulated, snr)
print(f'SNR: {snr}. sigma2: {sigma2}')

# для запуска с конкретными LLR:
llr_in = np.array([1.167170061968962, 1.6307143915761806, 2.5221245918395483,
 3.5512401343390816, 6.23346983681409, -0.2284348629800595, 9.211168143504485,
 0.31372871093553156, 6.007215069757603, 4.371153764272916,
 6.8730171124356705, 2.7507267411267975, 8.390081179083431, 8.288130867976705,
 5.856615789228111, 3.5640912311478057, 8.698797228217416, 8.869940523584562,
 8.994751678603668 ,3.7178026910527646, 0.8139102034041645 ,6.862437243759427,
 6.639418524370494, 5.019212929129337, 5.32780289907566 ,4.471140634345783,
 7.683999248209993 ,5.537175185761929,6.698118491718326 ,1.8147796132177345,
 5.984500468336725 ,4.124931127917616])
#------------------------------------------------------------------------------------------------------------------------

# считаем кол-во ошибок на входе
input_vector = bpsk_demodulation(llr_in)
errors = 0
errors_indexcies = []
for j in range(N):
    if input_vector[j] != codeword_initial[j]:
        errors += 1
        errors_indexcies.append(j)

print(f'Количество ошибок: {errors}\n Позиции ошибок: {errors_indexcies}\n')

print(f'Входные LLR: {llr_in}\n\n')


sp = BP(llr_in, H, 1)
decoded_word = sp.decode()
print("Декодированное слово")
print(decoded_word)
print("Изначальное кодовое слово")
print(np.array(codeword_initial))
print("Получено правильное кодовое слово ", np.array_equal(decoded_word, codeword_initial))

