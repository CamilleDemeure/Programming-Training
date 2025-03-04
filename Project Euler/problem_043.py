from itertools import permutations


def has_divisibility_property(num_str):
    primes = [2, 3, 5, 7, 11, 13, 17]
    for i in range(7):
        if int(num_str[i+1:i+4]) % primes[i] != 0:
            return False
    return True


def problem_43():
    total = 0
    for perm in permutations('0123456789'):
        num_str = ''.join(perm)
        if has_divisibility_property(num_str):
            total += int(num_str)
    return total


print(problem_43())
