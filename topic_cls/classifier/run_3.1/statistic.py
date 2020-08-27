import csv
from collections import Counter
import json
import sys
from os import path


def statistic(FILE_PATH):
    data = list(csv.reader(open(FILE_PATH,'rt')))
#print(data[0])
    labels = [row[0].strip().split('\t')[2] for row in data]
    counter = Counter(labels)
    print(counter.most_common())
    total = 0
    for idx, value in counter.most_common():
        total+=value

    print(total)
    for idx, value in counter.most_common():
        print('(%s, %f)'%(idx, value/total))
    return total, counter

if __name__=='__main__':
    FILE_PATH = path.join('data',sys.argv[1]) # input argument should be in ['train_idx','test_idx','valid_idx']
    total,counter = statistic(FILE_PATH)
