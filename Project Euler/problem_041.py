from itertools import permutations
from sympy import isprime


def problem_41():
    # 8- and 9-digit pandigitals are always divisible by 3
    for n in range(7, 0, -1):
        pandigitals = sorted([''.join(p)
                              for p
                              in permutations(''.join(map(str,
                                                          range(1, n+1))))],
                             reverse=True)
        for num in pandigitals:
            if isprime(int(num)):
                return int(num)


print(problem_41())
