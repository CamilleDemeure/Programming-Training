def problem_2(limit=4000000):
    fib_list = [1, 2]
    while fib_list[-1] + fib_list[-2] < limit:
        fib_list.append(fib_list[-1] + fib_list[-2])
    return sum([i for i in fib_list if i % 2 == 0])


print(problem_2())
