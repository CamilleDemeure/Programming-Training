from sympy import isprime
from itertools import combinations


def problem_51():
    num_digits = 6
    while True:
        primes = [p
                  for p in range(10**(num_digits-1), 10**num_digits)
                  if isprime(p)]
        prime_set = set(primes)

        for prime in primes:
            str_prime = str(prime)
            for num_replacements in range(1, len(str_prime)):
                for indices in combinations(range(len(str_prime)),
                                            num_replacements):
                    family = []
                    for digit in '0123456789':
                        new_prime = list(str_prime)
                        for index in indices:
                            new_prime[index] = digit
                        new_prime = int("".join(new_prime))
                        if ((new_prime in prime_set) and
                           (len(str(new_prime)) == num_digits)):
                            family.append(new_prime)
                    if len(family) == 8:
                        return min(family)
        num_digits += 1


print(problem_51())
