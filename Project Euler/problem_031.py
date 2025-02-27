def problem_031():
    goal = 200

    count_possibilities = [1] + [0] * goal

    for coin in [1, 2, 5, 10, 20, 50, 100, 200]:
        for i in range(len(count_possibilities) - coin):
            count_possibilities[i + coin] += count_possibilities[i]

    return count_possibilities[-1]


print(problem_031())
