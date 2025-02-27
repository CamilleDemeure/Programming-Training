from itertools import permutations


def is_valid_combination(multiplicand, multiplier, product):
    combined_str = f"{multiplicand}{multiplier}{product}"
    return len(combined_str) == 9 and set(combined_str) == set("123456789")


def problem_32():
    products = set()
    for perm in permutations("123456789"):
        perm_str = "".join(perm)

        # Case 1: 1-digit * 4-digit = 4-digit
        multiplicand = int(perm_str[0])
        multiplier = int(perm_str[1:5])
        product = int(perm_str[5:])
        if multiplicand * multiplier == product:
            products.add(product)

        # Case 2: 2-digit * 3-digit = 4-digit
        multiplicand = int(perm_str[0:2])
        multiplier = int(perm_str[2:5])
        product = int(perm_str[5:])
        if multiplicand * multiplier == product:
            products.add(product)

    return sum(products)


print(problem_32())
