def problem_56():
    return max([sum(int(i) for i in str(a**b))
                for a in range(100)
                for b in range(100)])


print(problem_56())
