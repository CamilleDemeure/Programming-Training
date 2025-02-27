def sum_of_digits(n):
    return sum([int(i) ** 5 for i in str(n)])


def problem_30():
    return sum([i for i in range(2, 1000000) if sum_of_digits(i) == i])


print(problem_30())
