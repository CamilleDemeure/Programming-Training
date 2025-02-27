import math


def problem_33():
    num = 1
    den = 1

    for d in range(10, 100):
        for n in range(10, d):

            n_left = n // 10
            n_right = n % 10
            d_left = d // 10
            d_right = d % 10

            if ((n_left == d_right and n_right * d == n * d_left) or
               (n_right == d_left and n_left * d == n * d_right)):
                num *= n
                den *= d

    return den // math.gcd(num, den)


print(problem_33())
