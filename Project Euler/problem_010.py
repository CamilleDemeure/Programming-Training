def problem_10(limit=2000000):
    sieve = [True] * limit
    sieve[0] = sieve[1] = False
    for start in range(2, int(limit ** 0.5) + 1):
        if sieve[start]:
            for multiple in range(start * start, limit, start):
                sieve[multiple] = False
    return sum(i for i, prime in enumerate(sieve) if prime)


print(problem_10())
