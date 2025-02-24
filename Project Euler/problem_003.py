def problem_3(n=600851475143):
    while True:
        d = smallest_divider(n)
        if d == n:
            return n
        else:
            n //= d


def smallest_divider(x):
    assert x >= 2
    d = 2
    while x % d != 0:
        d += 1
    return d


print(problem_3())
