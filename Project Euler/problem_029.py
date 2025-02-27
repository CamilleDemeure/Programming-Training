def problem_29():
    e = set(a ** b
            for a in range(2, 101)
            for b in range(2, 101)
            )
    return len(e)


print(problem_29())
