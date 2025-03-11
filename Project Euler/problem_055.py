def is_lychrel(n, max_iterations=50):
    def reverse_number(x):
        return int(str(x)[::-1])

    for _ in range(max_iterations):
        n = n + reverse_number(n)
        if str(n) == str(n)[::-1]:
            return False
    return True


def problem_55(limit=10000):
    return sum(1 for n in range(1, limit) if is_lychrel(n))


print(problem_55())
