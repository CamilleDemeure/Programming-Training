def length_chain(n):
    count = 0
    while n != 1:
        if n % 2 == 0:
            n //= 2
        else:
            n = 3 * n + 1
        count += 1
    return count


def problem_14():
    max_length = 0
    res = 1
    for i in range(2, 10 ** 6):
        lc = length_chain(i)
        if lc > max_length:
            max_length = lc
            res = i
    return res


print(problem_14())
