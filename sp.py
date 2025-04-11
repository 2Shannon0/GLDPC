import numpy as np
# L = np.array([-0.8,0.8,1.6,2.4,-2,-4.4,-1.6,-4.8])
from belief_propagation import BP

L = np.array([10.95476016,  5.35167048, -20.79616607,  6.05814424, 15.70320747,  7.02117129,
  7.83533787, 13.13784848,  5.04063348,  5.61883676,  7.71254401,  6.25898121,
  8.45298711, 10.81094375,  5.49060654])
def print_matrix(m):
    for i in range(m.shape[0]):
      for j in range(m.shape[1]):
          print("{:>15}".format(round(m[i][j], 5)), end=' ')
      print()


      
def f(x):
  return np.log(np.tanh(x/2))

def sumProduct(L, H, maxIter):
  H_gamma = np.copy(H)
  H_q = np.copy(H)
  out_L = np.zeros(H.shape[1])

  # Шаг 0 – инициализация значений, пришедших из канала
  for i in range(H.shape[0]):
    for j in range(H.shape[1]):
      H_q[i,j] = H[i,j] * L[j]

  # Начало итеративного декодирования
  for iter in range(maxIter):
    # Шаг 1 – передача сообщений от проверочных узлов символьным
    for i in range(H.shape[0]):
      for j in range(H.shape[1]):
        if H[i,j] == 1:
          indexes = np.setdiff1d(np.nonzero(H[i,:]), j)
          H_gamma[i,j] = -np.prod(np.sign(H_q[i,indexes]),dtype=float) * f(np.sum(np.abs(f(np.abs(H_q[i,indexes])))))

    # Шаг 2 – передача сообщений от символьных узлов проверочным     
    for i in range(H.shape[0]):
      for j in range(H.shape[1]):
        if H[i,j] == 1:
          indexes = np.setdiff1d(np.nonzero(H[:,j]), i)
          H_q[i,j] = L[j] + np.sum(H_gamma[indexes,j])  
    for i in range(H.shape[1]):
      out_L[i] = L[i] + np.sum(H_gamma[:,i])

    # Шаг 3 – формирование синдрома и попытка декодирования
    x_hat = np.array(out_L<0, dtype=int)
    
    if np.sum(np.matmul(x_hat, (H.T)) % 2) == 0: 
      x = x_hat
      return x
 

  
  
  

  # indexes = np.delete(np.nonzero(H[1]), 2)
  # print(np.nonzero(H[1]))
  # print(indexes)
  # # print(np.sign(H_q[0,indexes]))
  # # print(np.prod(np.sign(H_q[0,indexes])))
  # print((H_q[0,indexes]))
  # # print(-np.prod(np.sign(H_q[0,indexes])) * f(np.sum(np.abs(f(np.abs(H_q[0,indexes]))))))





# H = np.array([[1,1,1,0,0,0,0,0],
#               [0,0,0,1,1,1,0,0],
#               [1,0,0,1,0,0,1,0],
#               [0,1,0,0,1,0,0,1]],dtype=float)

H = np.array([[0,0,1,0,1,1,0,1,0,0,1,0,0,1,0],
              [1,0,0,0,1,1,1,0,1,0,0,0,0,1,0],
              [1,1,0,0,1,0,1,1,0,0,1,0,0,0,0],
              [1,0,0,0,1,1,1,1,0,0,1,0,0,1,1],

              [0,0,0,0,0,0,1,1,0,1,0,0,0,1,0],
              [0,0,0,1,1,0,1,0,0,0,1,0,0,0,0],
              [0,0,0,0,1,1,0,1,0,0,0,1,0,0,0],
              [1,0,0,0,1,0,0,0,0,0,0,0,1,1,0],

              
              ],dtype=float)

decoder = BP(L, H, 100)
# print('H')
# print_matrix(H)
print(sumProduct(L,H,100))
print(decoder.decode())
# assert(sumProduct(L,H,100) == decoder.decode())