from sympy import isprime


def problem_58(threshold=0.1):
    side_length = 1
    num_diagonals = 1
    num_primes = 0
    n = 1

    while True:
        side_length += 2
        for _ in range(4):
            n += side_length - 1
            if isprime(n):
                num_primes += 1

        num_diagonals += 4

        if num_primes / num_diagonals < threshold:
            return side_length


print(problem_58())
