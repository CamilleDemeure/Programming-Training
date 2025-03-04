def triangle(n):
    return n * (n + 1) // 2


def is_pentagonal(x):
    from math import sqrt
    n = (1 + sqrt(1 + 24 * x)) / 6
    return n.is_integer()


def is_hexagonal(x):
    from math import sqrt
    n = (1 + sqrt(1 + 8 * x)) / 4
    return n.is_integer()


def problem_45():
    n = 286
    while True:
        Tn = triangle(n)
        if is_pentagonal(Tn) and is_hexagonal(Tn):
            return Tn
        n += 1


print(problem_45())
