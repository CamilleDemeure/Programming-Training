from sympy import factorint


def problem_47(n):
    count = 0
    num = 2
    while True:
        if len(factorint(num)) == n:
            count += 1
            if count == n:
                return num - n + 1
        else:
            count = 0
        num += 1


print(problem_47(4))
