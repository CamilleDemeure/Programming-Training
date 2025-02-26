def fac(n):
    if n == 1 or n == 0:
        return 1
    else:
        return n*fac(n - 1)


def problem_20():
    return sum([int(i) for i in str(fac(100))])


print(problem_20())
