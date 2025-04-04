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

# codeword_initial = np.array([0] * N, dtype=int)
codeword_initial = np.array([0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

print(f'Кодовое слово: {codeword_initial}\n\n')

codeword_modulated = bpsk_modulation(codeword_initial)
print(f'Кодовое слово BPSK: {codeword_modulated}\n\n')

#------------------------------------Cоздание входных LLR----------------------------------------------------------------
snr = 4
llr_in, sigma2 = awgn_llr(codeword_modulated, snr)
print(f'SNR: {snr}. sigma2: {sigma2}')

# для запуска с конкретными LLR:
llr_in = np.array([12.884417162496169, -6.2033050457924395, -12.183548957787767,
 7.8392340500384465 ,-10.650885811311488, 10.180569466278087,
 11.73814628025913, -11.817917656487129, 7.028586948498691,
 11.952957903111392, 11.873484671982933, 2.737753098741238, 9.61758922509449,
 14.090998718911944 ,23.354159942767485 ,14.275775520660074,
 -18.429484806889352, -7.088357057410139 ,-16.08896482847936,
 -12.600270894895896 ,-14.882760325044694 ,-12.76080581058045,
 -13.547490045825905, -16.687968787413006 ,-13.194949048688752,
 -17.73545506098281 ,-13.996146928925599, -5.5742812251923475,
 -6.319479845214131 ,-18.828977347297567 ,-13.29129652463194,
 -8.69579021027348])
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


sp = BP(llr_in, H, 20)
decoded_word = sp.decode()
print("Декодированное слово")
print(decoded_word)
print("Изначальное кодовое слово")
print(np.array(codeword_initial))
print("Получено правильное кодовое слово ", np.array_equal(decoded_word, codeword_initial))

