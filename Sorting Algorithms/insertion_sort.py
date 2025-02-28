def insertion_sort(list_to_sort):
    n = len(list_to_sort)
    if n <= 1:
        return list_to_sort
    for i in range(1, n):
        key = list_to_sort[i]
        j = i - 1
        while j >= 0 and key < list_to_sort[j]:
            list_to_sort[j + 1] = list_to_sort[j]
            j -= 1
        list_to_sort[j + 1] = key
    return list_to_sort


print(insertion_sort([1, 6, 7, 9, 5, 4, 3, 8, 2]))

# The complexity of this sorting algorithm is O(n²).
