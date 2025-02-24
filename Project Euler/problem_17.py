def problem_17():
    return sum(len(convert(i)) for i in range(1, 1001))


def convert(n):
    if 0 <= n < 20:
        return units[n]
    elif 20 <= n < 100:
        return tens[n // 10] + (units[n % 10] if (n % 10 != 0) else "")
    elif 100 <= n < 1000:
        return (units[n // 100] +
                "hundred" +
                (("and" + convert(n % 100))
                 if (n % 100 != 0)
                 else ""))
    elif 1000 <= n < 1000000:
        return (convert(n // 1000) +
                "thousand" +
                (convert(n % 1000)
                 if (n % 1000 != 0)
                 else ""))
    else:
        raise ValueError()


units = [
       "zero",
       "one",
       "two",
       "three",
       "four",
       "five",
       "six",
       "seven",
       "eight",
       "nine",
       "ten",
       "eleven",
       "twelve",
       "thirteen",
       "fourteen",
       "fifteen",
       "sixteen",
       "seventeen",
       "eighteen",
       "nineteen"
       ]

tens = [
       "",
       "",
       "twenty",
       "thirty",
       "forty",
       "fifty",
       "sixty",
       "seventy",
       "eighty",
       "ninety"
       ]


print(problem_17())
