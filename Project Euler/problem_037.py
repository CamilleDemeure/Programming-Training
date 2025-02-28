from sympy import isprime


def is_truncatable_prime(n):
    if n < 10:
        return False
    str_n = str(n)
    return all(isprime(int(str_n[i:])) for i in range(len(str_n))) and \
        all(isprime(int(str_n[:i])) for i in range(1, len(str_n) + 1))


def problem_037(count=11):
    truncatable_primes = []
    num = 11
    while len(truncatable_primes) < count:
        if isprime(num) and is_truncatable_prime(num):
            truncatable_primes.append(num)
        num += 2
    return sum(truncatable_primes)


print(problem_037())
