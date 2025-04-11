import numpy as np
from bpsk import bpsk_modulation
from awgn import awgn_llr
from BCJR import BCJRDecoder
from GLDPC import GLDPC
from trellis4decoder import Trellis
from utils import read_csv

h_ldpc = read_csv('/Users/aleksejbandukov/Documents/python/GLDPC/matricies/H_LDPC(32,28).csv')
h_comp = read_csv('/Users/aleksejbandukov/Documents/python/GLDPC/matricies/H_ham(16,11).csv')

h_gldpc = read_csv('/Users/aleksejbandukov/Documents/python/GLDPC/matricies/H_gldpc_like_example.csv')

trellis1 = Trellis("/Users/aleksejbandukov/Documents/python/GLDPC/matricies/H_ham(16,11).csv")
trellis1.build_trellis()
code_component_decoder = BCJRDecoder(trellis1.edg)

N = h_ldpc.shape[1]
codeword_initial = np.array([0] * N, dtype=int)
codeword_modulated = bpsk_modulation(codeword_initial)

decoder = GLDPC(
    H_LDPC=h_ldpc,
    H_comp=h_comp,
    H_GLDPC=h_gldpc,
    CC_DECODER=code_component_decoder
)

llr_in, sigma2 = awgn_llr(codeword_modulated, 3)

print("Python")
print(decoder.decode(llr_in, sigma2, 10))

print("\nC++")
print(decoder.decode_cpp(llr_in, sigma2, 10))



