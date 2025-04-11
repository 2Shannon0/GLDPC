import numpy as np
from bpsk import bpsk_modulation
from awgn import awgn_llr
from BCJR import BCJRDecoder
from GLDPC import GLDPC
from trellis4decoder import Trellis
from utils import read_csv

h_ldpc = read_csv('/home/i17m5/GLDPC/matricies/LDPC_420_364.csv')
h_comp = read_csv('/home/i17m5/GLDPC/matricies/BCH_MATRIX_N_15_K_11_DEFAULT.csv')
h_gldpc = read_csv('/home/i17m5/GLDPC/matricies/H_gldpc_ALEXEI.csv')
trellis1 = Trellis("/home/i17m5/GLDPC/matricies/BCH_MATRIX_N_15_K_11_DEFAULT.csv")
# h_ldpc = read_csv('/home/i17m5/GLDPC/matricies/H_LDPC(32,28).csv')
# h_comp = read_csv('/home/i17m5/GLDPC/matricies/H_ham(16,11).csv')
# h_gldpc = read_csv('/home/i17m5/GLDPC/matricies/H_gldpc_1_like_example.csv')
# trellis1 = Trellis("/home/i17m5/GLDPC/matricies/H_ham(16,11).csv")

trellis1.build_trellis()
code_component_decoder = BCJRDecoder(trellis1.edg)

N = h_ldpc.shape[1]
# codeword_initial = np.array([0] * N, dtype=int)
codeword_initial = np.array([1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0])
# codeword_initial = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1])
codeword_modulated = bpsk_modulation(codeword_initial)

decoder = GLDPC(
    H_LDPC=h_ldpc,
    H_comp=h_comp,
    H_GLDPC=h_gldpc,
    CC_DECODER=code_component_decoder
)

llr_in, sigma2 = awgn_llr(codeword_modulated, 4)
print('sigma2: ', sigma2)

print("Python")
python_decoder_word = decoder.decode(llr_in, sigma2, 2)
print(python_decoder_word)

# error_indexes=[]
# for i in range(len(codeword_initial)):
#     if codeword_initial[i] != python_decoder_word[i]:
#         error_indexes.append(i)

# print(error_indexes)
# print(np.equal(codeword_initial, python_decoder_word))


print("\nC++")
c_decoder_word = decoder.decode_cpp(llr_in, sigma2, 2)
print(c_decoder_word)

print(np.equal(c_decoder_word, python_decoder_word))

# print(np.equal(codeword_initial, python_decoder_word))





