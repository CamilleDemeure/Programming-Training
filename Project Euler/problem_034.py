def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)


def problem_034():
    return sum([i
                for i in range(3, 10000000)
                if i == sum([factorial(int(j)) for j in str(i)])
                ])


print(problem_034())
