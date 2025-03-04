def problem_48():
    return str(sum(i ** i for i in range(1, 1001)))[-10::]


print(problem_48())
