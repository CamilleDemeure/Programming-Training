def problem_40():
    s = ''
    i = 0
    while len(s) < 10**6:
        i += 1
        s += str(i)
    return (int(s[0]) * int(s[9]) * int(s[99]) * int(s[999]) *
            int(s[9999]) * int(s[99999]) * int(s[999999]))


print(problem_40())
