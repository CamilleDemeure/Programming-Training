def cut(list_to_cut):
    if list_to_cut == []:
        return ([], [])
    else:
        return ([x for i, x in enumerate(list_to_cut) if i % 2 == 0],
                [x for i, x in enumerate(list_to_cut) if i % 2 == 1]
                )


def merge(l1, l2):
    if l1 == []:
        return l2
    elif l2 == []:
        return l1
    elif l1[0] < l2[0]:
        return l1[:1] + merge(l1[1:], l2)
    else:
        return l2[:1] + merge(l2[1:], l1)


def merge_sort(list_to_sort):
    if list_to_sort == [] or len(list_to_sort) == 1:
        return list_to_sort
    else:
        l1, l2 = cut(list_to_sort)
        return merge(merge_sort(l1), merge_sort(l2))


print(merge_sort([1, 6, 7, 9, 5, 4, 3, 8, 2]))

# With this algorithm we achieve a complexity of O(nlog(n))
