import datetime


def problem_19():
    return sum(1
               for y in range(1901, 2001)
               for m in range(1, 13)
               if datetime.date(y, m, 1).weekday() == 6)


print(problem_19())
