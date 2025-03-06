from sympy import primerange


def problem_50(limit=1000000):
    primes = list(primerange(1, limit))
    max_length = 0
    max_prime = 0

    for i in range(len(primes)):
        for j in range(i + max_length, len(primes)):
            total = sum(primes[i:j])
            if total > limit:
                break
            if total in primes:
                max_length = j - i
                max_prime = total

    return max_prime


print(problem_50())
