def pentagonal(n):
    return n * (3 * n - 1) // 2


def is_pentagonal(x):
    from math import sqrt
    n = (1 + sqrt(1 + 24 * x)) / 6
    return n.is_integer()


def problem_44():
    pentagonals = set()
    n = 1
    while True:
        Pn = pentagonal(n)
        for Pj in pentagonals:
            if (Pn - Pj) in pentagonals and is_pentagonal(Pn + Pj):
                return Pn - Pj
        pentagonals.add(Pn)
        n += 1


print(problem_44())
