def is_pandigital(s):
    return len(s) == 9 and set(s) == set("123456789")


def problem_038():
    largest = 0
    for num in range(1, 10000):
        concatenated = ""
        multiplier = 1
        while len(concatenated) < 9:
            concatenated += str(num * multiplier)
            multiplier += 1
        if len(concatenated) == 9 and is_pandigital(concatenated):
            largest = max(largest, int(concatenated))
    return largest


print(problem_038())
