def problem_23():
    m = 28124
    divisorsum = [0] * m
    for i in range(1, m):
        for j in range(i * 2, m, i):
            divisorsum[j] += i
    abundantnums = [i for (i, x) in enumerate(divisorsum) if x > i]

    expressible = [False] * m
    for i in abundantnums:
        for j in abundantnums:
            if i + j < m:
                expressible[i + j] = True

    return sum(i for i in range(len(expressible)) if not expressible[i])


print(problem_23())
