# WARNING: working code
# but time complexity too big to run on H = 1234567 and W = 123

def mex(s):
    m = 0
    while m in s:
        m += 1
    return m


def compute_grundy(w, h, dp):
    if dp[w][h] != -1:
        return dp[w][h]

    grundy_set = set()

    for x in range(1, w):
        for y in range(1, h):
            grundy_set.add(dp[x][y] ^
                           dp[w - x][y] ^
                           dp[x][h - y] ^
                           dp[w - x][h - y])

    dp[w][h] = mex(grundy_set)
    return dp[w][h]


def count_winning_cuts(w, h, dp):
    count = 0
    for x in range(1, w):
        for y in range(1, h):
            if dp[x][y] ^ dp[w - x][y] ^ dp[x][h - y] ^ dp[w - x][h - y] == 0:
                count += 1
    return count


def problem_933(W, H):
    dp = [[-1] * (H + 1) for _ in range(W + 1)]
    for w in range(W + 1):
        for h in range(H + 1):
            if w == 1 and h == 1:
                dp[w][h] = 0
            else:
                compute_grundy(w, h, dp)

    return sum(count_winning_cuts(w, h, dp)
               for w in range(2, W + 1)
               for h in range(2, H + 1))


W, H = 12, 123
result = problem_933(W, H)
print(f"D({W}, {H}) = {result}")
