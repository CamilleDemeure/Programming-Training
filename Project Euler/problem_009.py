def problem_9(n=1000):
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            c = n - b - a
            if a**2 + b**2 == c**2:
                return (a * b * c)


print(problem_9())
