from bpsk import bpsk_modulation, bpsk_demodulation
from awgn import awgn_llr
# from belief_propagation import BP
from utils import *
from trellis_repo import get_trellis
from BCJR import BCJRDecoder
# from GLDPC_debug_copy import GLDPC
from GLDPC import GLDPC


h_ldpc = read_csv('/home/i17m5/GLDPC/matricies/LDPC(420,196).csv')
h_comp = read_csv('/home/i17m5/GLDPC/matricies/BCH_MATRIX_N_15_K_11_DEFAULT.csv')

h_gldpc = read_csv('/home/i17m5/GLDPC/matricies/H_gldpc from_LDPC(420,196).csv')



N = h_ldpc.shape[1]
# print('h_ldpc\n')
# print_matrix(h_ldpc)

# Создаем декодер кода компонента
# Подгуржаем решетку
trellis1 = get_trellis('/home/i17m5/GLDPC/trellis_binaries/BCH_MATRIX_N_15_K_11_DEFAULT')
code_component_decoder = BCJRDecoder(trellis1.edg)

codeword_initial = np.array([0] * N, dtype=int)
# codeword_initial = np.array([0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

print(f'Кодовое слово: {codeword_initial}\n\n')

codeword_modulated = bpsk_modulation(codeword_initial)
print(f'Кодовое слово BPSK: {codeword_modulated}\n\n')

#------------------------------------Cоздание входных LLR----------------------------------------------------------------
snr = -2
llr_in, sigma2 = awgn_llr(codeword_modulated, snr)
print(f'SNR: {snr}. sigma2: {sigma2}')

# для запуска с конкретными LLR:
# llr_in = np.array([10.950952833565214 ,-10.678659273586316, -14.923683579591696,
#  14.452781232678571, -60.929935648832439 ,15.785517592718364,
#  9.615763268529186 ,-10.78005435292438,9.973346054136995 ,10.077513584546661,
#  8.213853639269018 ,16.21054050626045 ,10.867707520852655, 8.818863076771962,
#  2.5035848316290643 ,90.508225936535146,-11.197007059868271,
#  -14.92977092850371,-80.062282590486147 ,-13.23020354271099,
#  -14.531534922392634, -30.7859844582165256 ,-11.178678651524343,
#  -90.000984281999317,-60.646455387234506 ,-19.75275855549868,
#  -30.9992385268927255, -10.693459778814443 ,-11.523370784481257,
#  -90.649947835205834 ,-80.569103370018432 ,-14.942585974981771])
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
decoded_word = decoder.decode(llr_in, sigma2, 20)
print("Декодированное слово")
print(decoded_word)
print("Изначальное кодовое слово")
print(np.array(codeword_initial))
print("Получено правильное кодовое слово ", np.array_equal(decoded_word, codeword_initial))

