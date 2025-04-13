import numpy as np
# from belief_propagation import BP
from bpsk import bpsk_modulation, bpsk_demodulation
from awgn import awgn_llr
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from trellis_repo import get_trellis
from BCJR import BCJRDecoder
# from GLDPC_debug_copy import GLDPC
from GLDPC import GLDPC
from trellis4decoder import Trellis
from utils import read_csv

h_ldpc = read_csv('/home/i17m5/GLDPC/matricies/LDPC_420_364.csv')
h_comp = read_csv('/home/i17m5/GLDPC/matricies/BCH_MATRIX_N_15_K_11_DEFAULT.csv')

h_gldpc = read_csv('/home/i17m5/GLDPC/matricies/H_GLDPC_from_LDPC(420,364)_BCH(15,11).csv')

ESNO_START = 3
ESNO_END = 3
ESNO_STEP = 0.05
WRONG_DECODING_NUMBER = 120
N =h_ldpc.shape[1]


TITLE = f'Decoding GLDPC, LDPC(420,364) with BCH_MATRIX_N_15_K_11 WRONG_DECODING_NUMBER = {WRONG_DECODING_NUMBER}, ESNO_END = {ESNO_END} 1 fix iter c'
print('\n',TITLE,'\n')

# Создаем декодер кода компонента
# Подгуржаем решетку
# Раскоментить, если нет закэшированной решетки
trellis1 = Trellis("/home/i17m5/GLDPC/matricies/BCH_MATRIX_N_15_K_11_DEFAULT.csv")
trellis1.build_trellis()
# trellis1 = get_trellis('/home/i17m5/GLDPC/trellis_binaries/H_ham(16,11)')
code_component_decoder = BCJRDecoder(trellis1.edg)

# Задаем кодовое слово
# codeword_initial = np.array([0] * N, dtype=int)
# codeword_initial = np.array([0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])#1

# Альтернативное кодовое слово
# codeword_initial = np.array([0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0])#2
# codeword_initial = np.array([0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) #3

# codeword_initial = np.array([0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]) #4
# codeword_initial = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]) #5
# codeword_initial = np.array([0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) #6
# codeword_initial = np.array([1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]) #7
# codeword_initial = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0]) #8
codeword_initial = np.array([1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0])
codeword_modulated = bpsk_modulation(codeword_initial)

# Задаем список EsNo
esno_array = []
value = ESNO_START
while round(value, 2) <= ESNO_END:
    esno_array.append(round(value, 2))
    value += ESNO_STEP

fer = [0] * len(esno_array)
ber = [0] * len(esno_array)

decoder = GLDPC(
    H_LDPC=h_ldpc,
    H_comp=h_comp,
    H_GLDPC=h_gldpc,
    CC_DECODER=code_component_decoder
)

# # Запускаем моделирование
for (i, esno) in enumerate(esno_array):
    tests_passed, wrong_decoding, errors_at_all = 0, 0, 0

    print(f"\n-------------------- EsNo = {esno} --------------------")

    while wrong_decoding < WRONG_DECODING_NUMBER:
        tests_passed += 1

        # Для заданного отношения сигнал-шум считаем llr
        llr_in, sigma2 = awgn_llr(codeword_modulated, esno)

        # llr после декодирования
        # llr_out = BP(llr_in, H, 20)


        # Декодированное кодовое слово в бинарном виде
        # codeword_result = bpsk_demodulation(llr_out)
        codeword_result_py = decoder.decode(llr_in, sigma2, 3) # np.array([0] * N, dtype=int)
        codeword_result = decoder.decode(llr_in, sigma2, 3)
        # codeword_result = decoder.decode_cpp(llr_in=llr_in, sigma2=sigma2, max_iter=3, fix_iter=True)

        if not np.array_equal(codeword_result_py, codeword_result):
            print(np.equal(codeword_result_py, codeword_result))
            break


        # считаем кол-во ошибок
        errors = 0
        for j in range(N):
            if codeword_result[j] != codeword_initial[j]:
                errors += 1

        # если ошибки есть, то считаем fer & ber
        if errors != 0:
            wrong_decoding += 1
            errors_at_all += errors

            fer[i] = wrong_decoding / tests_passed
            ber[i] = errors_at_all / N / tests_passed

            print(f"fer = {fer[i]}, ber = {ber[i]}, tests_passed = {tests_passed}")

    print("\nRESULTS")
    print(esno_array)
    print(fer)
    print(ber)

fer_smooth = gaussian_filter1d(fer, sigma=2).tolist() # Параметр sigma овечает за то, насколько сильно сглаживать график. При 2 выглядит оптимально

plt.plot(esno_array, fer, label="Original", alpha=0.5, linewidth=1)
plt.plot(esno_array, fer_smooth, label="Smoothed", linewidth=2)
plt.yscale("log")  # Логарифмическая шкала по Y
plt.xlabel("EsNo")
plt.ylabel("FER")
plt.legend()
plt.grid(True, which="both", linestyle="--")
# plt.show()
plt.savefig(f'/home/i17m5/GLDPC/modeling_results/{TITLE}.png', dpi=300, bbox_inches='tight')
