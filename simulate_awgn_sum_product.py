import numpy as np
from belief_propagation import BP
from bpsk import bpsk_modulation, bpsk_demodulation
from awgn import awgn_llr
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from trellis_repo import get_trellis
from BCJR import BCJRDecoder

from utils import read_csv

ESNO_START = 5.2
ESNO_END = 6
ESNO_STEP = 0.2
WRONG_DECODING_NUMBER = 30
# H = read_csv('/home/i17m5/GLDPC/matricies/LDPC_2080_1760.csv')
H = read_csv('/home/i17m5/GLDPC/matricies/H_GLDPC_from_LDPC(420,364)_BCH(15,11).csv')

N = H.shape[1]


TITLE = f'Decoding H_GLDPC_from_LDPC(420,364)_BCH(15,11) by SP, WRONG_DECODING_NUMBER = {WRONG_DECODING_NUMBER}, ESNO_END = {ESNO_END}'
print('\n',TITLE,'\n')


# Задаем кодовое слово
codeword_initial = np.array([0] * N, dtype=int)
# codeword_initial = np.array([0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])


# codeword_initial = np.array([0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0])

codeword_modulated = bpsk_modulation(codeword_initial)

# Задаем список EsNo
esno_array = []
value = ESNO_START
while round(value, 2) <= ESNO_END:
    esno_array.append(round(value, 2))
    value += ESNO_STEP

# Создаем fer & ber
fer = [0] * len(esno_array)
ber = [0] * len(esno_array)

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
        codeword_result = BP(llr_in, H, 20).decode()


        # считаем кол-во ошибок
        errors = 0
        if codeword_result is None:
            errors+=1
        else:
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
