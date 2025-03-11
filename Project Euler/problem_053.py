import numpy as np


def problem_53(size=101):
    C = np.zeros((size, size), dtype=object)
    C[0][0] = 1

    for i in range(1, size):
        C[i][0] = 1
        for j in range(1, i + 1):
            C[i][j] = C[i - 1][j] + C[i - 1][j - 1]

    return sum(1
               for i in range(size)
               for j in range(size)
               if C[i][j] > 1000000)


print(problem_53())
