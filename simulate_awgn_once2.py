from bpsk import bpsk_modulation, bpsk_demodulation
from awgn import awgn_llr
# from belief_propagation import BP
from utils import *
from trellis_repo import get_trellis
from BCJR import BCJRDecoder
from GLDPC import GLDPC

h_ldpc = read_csv('/home/i17m5/GLDPC/matricies/H_LDPC(32,28).csv')
h_comp = read_csv('/home/i17m5/GLDPC/matricies/H_ham(16,11).csv')

h_gldpc = read_csv('/home/i17m5/GLDPC/matricies/H_GLDPC.csv')



N = h_ldpc.shape[1]
# print('h_ldpc\n')
# print_matrix(h_ldpc)

# Создаем декодер кода компонента
# Подгуржаем решетку
trellis1 = get_trellis('/home/i17m5/GLDPC/trellis_binaries/H_ham(16,11)')
code_component_decoder = BCJRDecoder(trellis1.edg)

codeword_initial = np.array([0] * N, dtype=int)
# codeword_initial = np.array([1,0,1,0,1,1,1,1])

print(f'Кодовое слово: {codeword_initial}\n\n')

codeword_modulated = bpsk_modulation(codeword_initial)
print(f'Кодовое слово BPSK: {codeword_modulated}\n\n')

#------------------------------------Cоздание входных LLR----------------------------------------------------------------
snr = 4
llr_in, sigma2 = awgn_llr(codeword_modulated, snr)
print(f'SNR: {snr}. sigma2: {sigma2}')

# для запуска с конкретными LLR:
# llr_in = np.array([0.8, 0.8, -3.6, 2.4, 2,-4.4,-1.6,-4.8])
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


# sp = BP(llr_in, H, 1)
# decoded_word = sp.decode()

decoder = GLDPC(
    H_LDPC=h_ldpc,
    H_comp=h_comp,
    H_GLDPC=h_gldpc,
    CC_DECODER=code_component_decoder
)
decoded_word = decoder.decode(llr_in, sigma2, 1)
print("Декодированное слово")
print(decoded_word)
print("Изначальное кодовое слово")
print(np.array(codeword_initial))
print("Получено правильное кодовое слово ", np.array_equal(decoded_word, codeword_initial))

