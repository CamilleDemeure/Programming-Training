def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def problem_7(n):
    count = 0
    num = 1
    while count < n:
        num += 1
        if is_prime(num):
            count += 1
    return num


print(problem_7(10001))
