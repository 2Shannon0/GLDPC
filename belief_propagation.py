import numpy as np
from utils import *

def print_matrix(m):
    for i in range(m.shape[0]):
      for j in range(m.shape[1]):
          print("{:>5}".format(round(m[i][j], 3)), end=' ')
      print()

class BP:
    def __init__(self, L, H, maxIter=20):
        self.L = L
        self.H = H
        self.maxIter = maxIter

    def decode(self):
      H_gamma = np.copy(self.H).astype(float)
      # print('\nH_gamma изначально:')
      # print_matrix(H_gamma)

      H_q = np.copy(self.H).astype(float)
      # print('\nH_q изначально:')
      # print_matrix(H_q)
      # print('\nllr_in: ', self.L)

      
      out_L = np.zeros(self.H.shape[1])
      # print('out_L')
      # print(out_L)


      for i in range(self.H.shape[0]):
        for j in range(self.H.shape[1]):
          H_q[i,j] = self.H[i,j] * self.L[j]
      # print('\nH_q после инициализации:')
      # print_matrix(H_q)

      for iter in range(self.maxIter):
        for i in range(self.H.shape[0]):
          for j in range(self.H.shape[1]):
            if self.H[i,j] == 1:
              indexes = np.setdiff1d(np.nonzero(self.H[i,:]), j)
              H_gamma[i,j] = -np.prod(np.sign(H_q[i,indexes]),dtype=float) * f(np.sum(np.abs(f(np.abs(H_q[i,indexes])))))
        # print('H_gamma')
        # print_matrix(H_gamma)

        for i in range(self.H.shape[0]):
          for j in range(self.H.shape[1]):
            if self.H[i,j] == 1:
              indexes = np.setdiff1d(np.nonzero(self.H[:,j]), i)
              H_q[i,j] = self.L[j] + np.sum(H_gamma[indexes,j])
        # print('H_q')
        # print_matrix(H_q)

        for i in range(self.H.shape[1]):
          out_L[i] = self.L[i] +  np.sum(H_gamma[:,i])

        x_hat = np.array(out_L<0, dtype=int)
        # print('Iter: ', iter)
        #ВЫВОДИТЬ КОДОВОЕ СЛОВО НА КАЖДОМ ШАГЕ
        # print('\n',x_hat)
        if np.sum(np.matmul(x_hat, (self.H.T)) % 2) == 0:
          x = x_hat
          # print("\nОшибки исправлены, кол-во итераций", iter+1)
          # print('\nout_L', out_L)
          return x
      # print("\nКодовое слово не найдено, кол-во итераций", iter+1)
      # print('\nout_L', out_L)