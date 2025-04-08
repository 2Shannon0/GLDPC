# common functions
import numpy as np
import csv

# Загружает CSV-файл в numpy-массив.
def read_csv( filename ):
    with open( filename, newline='') as f:
        reader = csv.reader(f)
        symbols = []
        for row in reader:
            rows = []
            for symbol in row:
                rows.append(int(symbol))
            symbols.append(rows)
    return np.array(symbols)

def save_to_csv(matrix, filepath):
    np.savetxt(filepath, matrix, delimiter=',', fmt='%d')

def print_matrix(m):
    for i in range(m.shape[0]):
      for j in range(m.shape[1]):
          print(int(m[i][j]), end=" ")
      print()

def print_matrixA(m):
    for i in range(m.shape[0]):
      print('[', end = ' ')
      for j in range(m.shape[1]):
          print(int(m[i][j]), end=" ")
      print(']')
      print()

def print_matrix_for_c(m):
    for i in range(m.shape[0]):
      print('{', end = '')
      for j in range(m.shape[1]):
          if j != m.shape[1] - 1:
              print(int(m[i][j]), end=", ")
          else:
              print(int(m[i][j]), end="")
      if i != m.shape[0] - 1:
          print('}, ')
      else:
          print("}")
      print()

def create_matrix_from_basis(basis, n, k):
    result = np.zeros([n - k, n])
    result[0] = result[0] + basis
    for i in range(1, n - k):
        result[i] = result[i] + shifting(result[i - 1])
    return result

def create_extended_matrix_from_basis(basis):
    size = basis.size
    result = np.zeros([size, size])
    result[0] = result[0] + basis
    for i in range(1, size):
        result[i] = result[i] + shifting(result[i - 1])
    return result

def create_spread_matrix(h_r, count):
    size = h_r.shape[0]
    hasOne = False
    iterations = 0
    max_iter = size * 3

    while not hasOne:
        if iterations == max_iter:
            raise ValueError(f"Невозможно заданную матрицу разбить на {count}.")

        matrices = []
        for i in range(count):
            matrices.append(np.zeros([size, size]))
        for i in range(size):
            for j in range(size):
                r = np.random.randint(count)
                matrices[r][i][j] = h_r[i][j]

        hasOne = True
        for m in matrices:
            hasOne *= has_one_in_rows(m)
            if not hasOne:
                break

        iterations += 1

    result = matrices[0]
    for i in range(1, count):
        result = np.concatenate((result, matrices[i]), axis=1)

    return result

def shifting(a):
    length = a.size
    result = np.zeros(length)
    for i in range(length):
        result[(i + 1) % length] = a[i]
    return result

def has_one_in_rows(m):
    m_t = np.transpose(m)
    for i in range(m_t.shape[0]):
        if 1 not in m_t[i]:
            return False
    return True

def create_max_spread_matrix(h_r):
    count = 2
    result = create_spread_matrix(h_r, count)
    while True:
        count += 1
        try:
            result = create_spread_matrix(h_r, count)
        except ValueError:
            count -= 1
            break

    print(f"Максимальное кол-во матриц для разбиения {count}, плотность Hs {calc_density(result)}")
    return result

def create_random_code_word(k):
    return np.round(np.random.rand(k))

def calc_density(a):
    return np.sum(a) / (a.shape[0] * a.shape[1])

def reverse(a):
    return (a + 1) % 2

def rearrange_code(m, c):
    d = int(m.shape[1] / m.shape[0])

    codes = []
    for i in range(d):
        codes.append(c)

    result = codes[0]
    for i in range(1, d):
        result = np.concatenate((result, codes[i]))

    return result


def awgn(c, sigma=0.5):
    # rewrite
    # 1 -> -1
    # 0 -> +1
    c1 = c * -2 + 1

    #AWGN
    loc = 0,
    sigma = np.sqrt(sigma)
    size = len(c1)
    e = np.array(np.random.normal(loc, sigma, size)) # 2y / sigma^2

    c2 = c1+e

    # rewrite vector with errors in -1 +1
    c3 = []

    for i in c2:
        if i < 0: i = -1
        if i > 0: i = 1
        c3.append(i)
    c3 = np.array(c3)

    #determining the number of errors
    # def findNumberOfE(a,b):
    # k = 0
    # for i in c1:
    #   if c1[i] == c3[i]: k += 1
    errorIndexcies = []
    for i in range(c3.shape[0]):
        if c3[i] != c1[i]: errorIndexcies.append(i)

    cLLR =[]
    for i in c2:
        i = 2*i/(sigma**2)
        cLLR.append(i)
    cLLR = np.array(cLLR)

    return cLLR, errorIndexcies


# def f(x):
#     return np.log(np.tanh(x/2))
# та же функция но с подменой нулей на минимальные положительные значения
def f(x):
    # Предотвращаем log(0) и числовую нестабильность
    tanh_val = np.tanh(x / 2)
    tanh_val = np.where(tanh_val == 0, np.finfo(float).eps, tanh_val)
    return np.log(tanh_val)