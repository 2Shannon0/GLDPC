import numpy as np
from matplotlib import pyplot as plt

# Повторно определим функции и сгенерируем данные

n = 10000
snr_db = 5
codeword = np.random.choice([-1, 1], size=n)

def awgn_llr_complex_v1(codeword, snr_db):
    signal = np.array(codeword, dtype=complex)
    noiseVar = 10 ** (-snr_db / 10)
    noise = (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)) / np.sqrt(2)
    received_signal = signal + np.sqrt(noiseVar) * noise
    llr_values = 2 * received_signal.real / noiseVar
    return llr_values, received_signal, noise

def awgn_llr_complex_v2(codeword, snr_db):
    signal = np.array(codeword, dtype=complex)
    noiseVar = 10 ** (-snr_db / 10)
    real_imag = np.random.normal(loc=0, scale=np.sqrt(2)/2, size=(len(signal), 2))
    noise = real_imag.view(np.complex128)
    received_signal = signal + np.sqrt(noiseVar) * noise
    llr_values = 2 * received_signal.real / noiseVar
    return llr_values, received_signal, noise

# Генерируем данные
llr1, rx1, noise1 = awgn_llr_complex_v1(codeword, snr_db)
llr2, rx2, noise2 = awgn_llr_complex_v2(codeword, snr_db)

# Теперь можно строить графики
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Сравнение распределения шума (модуль)
axs[0, 0].hist(np.abs(noise1), bins=100, alpha=0.6, label='noise1 (randn)', density=True)
axs[0, 0].hist(np.abs(noise2), bins=100, alpha=0.6, label='noise2 (view)', density=True)
axs[0, 0].set_title('Распределение |noise|')
axs[0, 0].legend()
axs[0, 0].grid(True)

# Сравнение мощности шума
power1 = np.mean(np.abs(noise1)**2)
power2 = np.mean(np.abs(noise2)**2)
axs[0, 1].bar(['randn', 'view'], [power1, power2])
axs[0, 1].set_title('Средняя мощность шума')
axs[0, 1].grid(True)

# Сравнение LLR
axs[1, 0].hist(llr1, bins=100, alpha=0.6, label='LLR1 (randn)', density=True)
axs[1, 0].hist(llr2, bins=100, alpha=0.6, label='LLR2 (view)', density=True)
axs[1, 0].set_title('Распределение LLR')
axs[1, 0].legend()
axs[1, 0].grid(True)

# Scatter: сравнение LLR1 vs LLR2
axs[1, 1].scatter(llr1[:500], llr2[:500], alpha=0.5, s=10)
axs[1, 1].set_title('LLR: randn vs view (500 точек)')
axs[1, 1].set_xlabel('LLR1 (randn)')
axs[1, 1].set_ylabel('LLR2 (view)')
axs[1, 1].grid(True)

# plt.tight_layout()
plt.savefig('./test.png')
