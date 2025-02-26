from itertools import permutations


def problem_24():
    digits = "0123456789"
    perms = permutations(digits)
    millionth_perm = next(x
                          for i, x in enumerate(perms, start=1)
                          if i == 1_000_000
                          )
    return "".join(millionth_perm)


print(problem_24())
