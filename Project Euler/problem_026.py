def recurring_cycle_length(d):
    remainders = {}
    numerator = 1
    position = 0

    while numerator != 0:
        if numerator in remainders:
            return position - remainders[numerator]
        remainders[numerator] = position
        numerator = (numerator % d) * 10
        position += 1

    return 0


def problem_26():
    longest_cycle = 0
    best_d = 0

    for d in range(1, 1000):
        cycle_length = recurring_cycle_length(d)
        if cycle_length > longest_cycle:
            longest_cycle = cycle_length
            best_d = d

    return best_d


print(problem_26())
