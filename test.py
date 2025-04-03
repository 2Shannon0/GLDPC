import numpy as np
from GLDPC import GLDPC
from utils import read_csv

def print_matrix(m):
    for i in range(m.shape[0]):
      for j in range(m.shape[1]):
          print("{:>3}".format(round(m[i][j], 5)), end=' ')
      print()

H_ldpc = read_csv('/home/i17m5/GLDPC/matricies/H_LDPC(32,28).csv')
H_cc = read_csv('/home/i17m5/GLDPC/matricies/H_ham(16,11).csv')
H_g = read_csv('/home/i17m5/GLDPC/matricies/H_GLDPC.csv')
g1 = GLDPC(H_ldpc, H_cc)

print('\n H_GLDPC целиком')
print_matrix(g1.H_GLDPC)
