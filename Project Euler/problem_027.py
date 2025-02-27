import math


def is_prime(x: int) -> bool:
    if x <= 1:
        return False
    elif x <= 3:
        return True
    elif x % 2 == 0:
        return False
    else:
        for i in range(3, math.isqrt(x) + 1, 2):
            if x % i == 0:
                return False
        return True


def problem_27():
    max_consecutive_prime = 0
    for a in range(-1000, 1001):
        for b in range(-1000, 1001):
            n = 0
            while is_prime(n ** 2 + a * n + b):
                n += 1
            if n > max_consecutive_prime:
                max_consecutive_prime = n
                product_max = a * b
    return product_max


print(problem_27())
