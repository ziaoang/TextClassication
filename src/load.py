import csv

alphabet = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
alphabet_size = len(alphabet)
char2ind = {}
for i in range(alphabet_size):
    char2ind[alphabet[i]] = i

def load(name):
    if name == "AG":
        return load_AG()
    elif name == "DBP":
        return load_DBP()

def load_AG():
    class_num = 4
    train_set = load_base("../data/ag_news_csv/train.csv")
    test_set = load_base("../data/ag_news_csv/test.csv")
    return class_num, train_set, test_set

def load_DBP():
    class_num = 14
    train_set = load_base("../data/dbpedia_csv/train.csv")
    test_set = load_base("../data/dbpedia_csv/test.csv")
    return class_num, train_set, test_set

def load_base(file_path):
    data = []
    with open(file_path) as f:
        for label, title, desc in csv.reader(f):
            data.append([int(label)-1, text2fea(title + ' ' + desc)])
    return data

def text2fea(text, max_text_len=1014):
    text_len = len(text)
    fea = [-1] * max_text_len
    for i in range(min(text_len, max_text_len)):
        c = text[text_len-1-i]
        if c in char2ind:
            fea[i] = char2ind[c]
    return fea


