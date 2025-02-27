def problem_28(n=1001):
    if n % 2 == 0:
        raise ValueError("Spiral size must be an odd number.")

    total = 1
    for size in range(3, n + 1, 2):
        total += 4 * (size ** 2) - 6 * (size - 1)

    return total


print(problem_28())
