from sympy import isprime
from math import sqrt


def can_be_written_as_goldbach(n):
    for i in range(1, int(sqrt(n // 2)) + 1):
        if isprime(n - 2 * i * i):
            return True
    return False


def problem_46():
    n = 9
    while True:
        if not isprime(n) and not can_be_written_as_goldbach(n):
            return n
        n += 2


print(problem_46())
