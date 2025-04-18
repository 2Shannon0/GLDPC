import numpy as np

def awgn_llr_complex(codeword, snr_db):
    signal = np.array(codeword, dtype=complex)
    noiseVar = 10 ** (-snr_db / 10) # дисперсия sigma^2
    sigma2 = noiseVar

    noise = (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)) / np.sqrt(2)

    received_signal = signal + np.sqrt(sigma2) * noise

    llr_values = 2 * received_signal.real / sigma2

    return llr_values.astype(list), sigma2
