def partition(pivot, list_to_partition):
    return ([i for i in list_to_partition if i <= pivot],
            [i for i in list_to_partition if i > pivot]
            )


def quick_sort(list_to_sort):
    if list_to_sort == []:
        return list_to_sort
    else:
        l1, l2 = partition(list_to_sort[0], list_to_sort[1:])
        return (quick_sort(l1) + list_to_sort[:1] + quick_sort(l2))


print(quick_sort([1, 6, 7, 9, 5, 4, 3, 8, 2]))

# The mean complexity of this algorithm is O(nlog(n)).
# However in the worst case (if the list is already sorted),
# the max complexity is O(n²).
