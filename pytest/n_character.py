import _init_paths


def count_character(dire):
    with open(dire, 'r') as f:
        txt = f.read().split('\n')
    arr = []
    for row in txt:
        for c in row:
            if c not in arr:
                arr.append(c)
        if len(arr) == 24:
            break
    print(len(arr))
    return arr


if __name__ == '__main__':
    # data_dir = "/home/zhengbian/Dataset/uniref/uniref.txt"
    # arr = count_character(data_dir)
    # print(arr)
    arr = ['M', 'T', 'S', 'P', 'V', 'R', 'E', 'W', 'D', 'G', 'L', 'A', 'K', 'F', 'H', 'Y', 'I', 'Q', 'N', 'C', 'O', 'U',
           'Z', 'B']
    # ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y', 'Z']
    arr.sort()
    print(arr)
# count the character
