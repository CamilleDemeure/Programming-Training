def problem_57():
    res = 0
    numer = 0
    denom = 1
    for _ in range(1000):
        numer, denom = denom, denom * 2 + numer
        if len(str(numer + denom)) > len(str(denom)):
            res += 1
    return res


print(problem_57())
