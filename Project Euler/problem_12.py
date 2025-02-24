import math


def nb_divisors(n):
    end = math.isqrt(n)
    result = 0
    for i in range(1, end + 1):
        if n % i == 0:
            result += 2
    if end ** 2 == n:
        result -= 1
    return result


def problem_12():
    triangle = 0
    i = 1
    while nb_divisors(triangle) <= 500:
        triangle += i
        i += 1
    return str(triangle)


print(problem_12())
