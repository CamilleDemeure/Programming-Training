from itertools import permutations
from sympy import isprime


def problem_49():
    four_digit_primes = [n
                         for n in range(1000, 10000)
                         if isprime(n)]
    prime_set = set(four_digit_primes)

    for prime in four_digit_primes:
        perms = set(int("".join(p))
                    for p in permutations(str(prime)))
        prime_perms = sorted([p
                              for p in perms
                              if p in prime_set and p >= 1000])

        for i in range(len(prime_perms)):
            for j in range(i + 1, len(prime_perms)):
                k = 2 * prime_perms[j] - prime_perms[i]
                if k in prime_perms and k != 8147:
                    return "".join(map(str,
                                       [prime_perms[i],
                                        prime_perms[j],
                                        k]))


print(problem_49())
