import math


def list_primality(n):
    result = [True] * (n + 1)
    result[0] = result[1] = False
    for i in range(math.isqrt(n) + 1):
        if result[i]:
            for j in range(i * i, len(result), i):
                result[j] = False
    return result


list_prime = list_primality(999999)


def is_circular_prime(n):
    s = str(n)
    return all(list_prime[int(s[i:] + s[:i])] for i in range(len(s)))


def problem_035():
    ans = sum(1
              for i in range(len(list_prime))
              if is_circular_prime(i))
    return str(ans)


print(problem_035())
