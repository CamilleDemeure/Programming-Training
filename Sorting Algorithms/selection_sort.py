def selection_sort(list_to_sort):
    for i in range(len(list_to_sort)):
        min = i
        for j in range(i + 1, len(list_to_sort)):
            if list_to_sort[min] > list_to_sort[j]:
                min = j
        tmp = list_to_sort[i]
        list_to_sort[i] = list_to_sort[min]
        list_to_sort[min] = tmp
    return list_to_sort


print(selection_sort([1, 6, 7, 9, 5, 4, 3, 8, 2]))

# The complexity of this algorithm is O(n²).
