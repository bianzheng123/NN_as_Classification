def get_max_length(dire):
    with open(dire, 'r') as f:
        text = f.read().split('\n')[:-1]
        print(max([len(_) + 1 for _ in text]))


dire = "/home/zhengbian/Dataset/uniref/uniref.txt"
get_max_length(dire)
dire = "/home/zhengbian/Dataset/uniref/unirefquery.txt"
get_max_length(dire)
