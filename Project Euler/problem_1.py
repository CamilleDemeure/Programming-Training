def problem_1(n=1000):
    return sum([i for i in range(n) if (i % 5 == 0 or i % 3 == 0)])


print(problem_1())
