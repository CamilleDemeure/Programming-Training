def problem_039(limit=1000):
    max_solutions = 0
    best_p = 0

    for p in range(1, limit + 1):
        solutions = 0
        for a in range(1, p // 3):
            for b in range(a, (p - a) // 2):
                c = p - a - b
                if a * a + b * b == c * c:
                    solutions += 1

        if solutions > max_solutions:
            max_solutions = solutions
            best_p = p

    return best_p


print(problem_039(1000))
