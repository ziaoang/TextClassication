import csv

def load(name, alphabet, max_text_len):
    char2ind = {}
    for i in range(len(alphabet)):
        char2ind[alphabet[i]] = i
    
    if name == "AG":
        return load_AG(char2ind, max_text_len)
    elif name == "DBP":
        return load_DBP(char2ind, max_text_len)

def load_AG(char2ind, max_text_len):
    class_num = 4
    train_set = load_base("../data/ag_news_csv/train.csv", char2ind, max_text_len)
    test_set = load_base("../data/ag_news_csv/test.csv", char2ind, max_text_len)
    return class_num, train_set, test_set

def load_DBP(char2ind, max_text_len):
    class_num = 14
    train_set = load_base("../data/dbpedia_csv/train.csv", char2ind, max_text_len)
    test_set = load_base("../data/dbpedia_csv/test.csv", char2ind, max_text_len)
    return class_num, train_set, test_set

def load_base(file_path, char2ind, max_text_len):
    data = []
    with open(file_path) as f:
        for label, title, desc in csv.reader(f):
            data.append([int(label)-1, text2fea(title + ' ' + desc, char2ind, max_text_len)])
    return data

def text2fea(text, char2ind, max_text_len):
    text = text.lower()[::-1][:max_text_len]
    fea = [-1] * max_text_len
    for i in range(len(text)):
        if text[i] in char2ind:
            fea[i] = char2ind[text[i]]
    return fea



