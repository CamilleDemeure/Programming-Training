def is_palyndrom(n, base):
    s = format(n, 'b' if base == 2 else '')
    return s == s[::-1]


def problem_036():
    return sum(n
               for n in range(1000000)
               if is_palyndrom(n, 10) and is_palyndrom(n, 2)
               )


print(problem_036())
