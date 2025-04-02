from bpsk import bpsk_modulation, bpsk_demodulation
from awgn import awgn_llr
from belief_propagation import BP
from utils import *

H = read_csv('/home/i17m5/GLDPC/matricies/H_ham(8,4).csv')

N = H.shape[1]
print('H\n')
print_matrix(H)

# codeword_initial = np.array([0] * N, dtype=int)
codeword_initial = np.array([1,0,1,0,1,1,1,1])

print(f'Кодовое слово: {codeword_initial}\n\n')

codeword_modulated = bpsk_modulation(codeword_initial)
print(f'Кодовое слово BPSK: {codeword_modulated}\n\n')

#------------------------------------Cоздание входных LLR----------------------------------------------------------------
# snr = 2
# llr_in, sigma2 = awgn_llr(codeword_modulated, snr)
# print(f'SNR: {snr}. sigma2: {sigma2}')
# 
# для запуска с конкретными LLR:
llr_in = np.array([-0.8, 0.8, -3.6, 2.4, 2,-4.4,-1.6,-4.8])
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


sp = BP(llr_in, H, 10)
decoded_word = sp.decode()
print("Декодированное слово")
print(decoded_word)
print("Изначальное кодовое слово")
print(np.array(codeword_initial))
print("Получено правильное кодовое слово ", np.array_equal(decoded_word, codeword_initial))

