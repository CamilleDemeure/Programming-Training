def problem_26():
    x1 = 1
    x2 = 1
    count = 1
    while len(str(x1)) < 1000:
        x2, x1 = x1 + x2, x2
        count += 1
    return count


print(problem_26())
