# import numpy as np

# # Проверочная матрица
# H = np.array([
#     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#     [1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
#     [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0]
# ])

# # Используем numpy для поиска решения уравнения H * x = 0
# from sympy import Matrix
# H_sympy = Matrix(H)
# solution = H_sympy.nullspace()

# # Получаем решение
# codeword = np.array(solution[0]).flatten()  # Берем первое ненулевое решение

# # Проверим, что оно действительно кодовое (H * x = 0)
# # print("Ненулевое кодовое слово:", codeword)
# # check = np.dot(H, codeword) % 2  # Проверим, что H * x = 0 по модулю 2
# # print("Проверка, что оно кодовое (H * x % 2 = 0):", check)
# # codeword=[0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# codeword=[0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


# # codeword=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# if np.sum(np.matmul(codeword, (H.T)) % 2) == 0:
#     print("Ненулевое кодовое слово:", codeword)
# else:
#     print("Кодовое слово хуйня")
from BCJR import BCJRDecoder 
from trellis_repo import get_trellis

trellis1 = get_trellis('/home/i17m5/GLDPC/trellis_binaries/H_ham(16,11)')

llr_in=[ 14.45278123,  16.21054051 ,  8.81886308  , 2.50358483 , 90.50822594,
  15.78551759  , 9.61576327 ,-10.78005435  , 8.21385364,  10.07751358,
 -14.92368358 ,-10.67865927, -60.92993565 , 10.86770752,  10.95095283,
   9.97334605]
sigma2 =0.15848931924611134

decoder = BCJRDecoder(trellis1.edg)

print(decoder.decode(llr_in,sigma2))
