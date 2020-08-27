import csv
import json
import random
from sklearn.model_selection import train_test_split
import os 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',type=str,help='which dataset is used in Alexa topical dataset for testing, options can be train, valid_rare, valida_freq, test_freq, test_rare',required=True,choices=['train','valid_rare','valid_freq','test_freq','test_rare'])
args = parser.parse_args()

DATASET = args.dataset # enter the dataset name to process, select one from train, test_freq, test_rare, valid_freq, valid_rare

if DATASET=='train':
    os.system('rm -r ./data/')
    os.system('mkdir data/')
    #reader = csv.reader(open('/nas/home/zixiliu/new/Dialogue_System_Hackathon/topic_cls/heuristics/data/gathered_labels.tsv'), delimiter='\t')
    reader = csv.reader(open(os.path.join('../../heuristics/data','gathered_labels_'+DATASET+'.tsv')), delimiter='\t')
    data = []
    Y = []
#print(data[:2])
    train, test, general = [], [], []
    flag = True
    for row in reader:
        if 'rule_1' in row[2]:
            data.append(row)
            Y.append(json.loads(row[3])[0])
        elif 'agreement' in row[2]:
            data.append(row)
            Y.append(json.loads(row[3])[0])
        elif 'general' in row[2]:
            data.append(row)
            Y.append(json.loads(row[3])[0])

    X_train, test, y_train, y_test = train_test_split(
                data, Y, test_size=0.15, random_state=42)
    train, valid, y_tr, y_valid = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
            )
#random.shuffle(general)
#general1 = general[:int(len(general)*(len(train)/(len(train)+len(test))))]
#general2 = general[int(len(general)*(len(train)/(len(train)+len(test)))):]
#train = train+general1
#test = test + general2

#random.shuffle(train)
#valid_per = .3
#valid_idx = int(len(train) * .3)
    writer = csv.writer(open('./data/train_idx', 'wt'), delimiter='\t')
    for row in train:
        writer.writerow((row[0], row[1], json.loads(row[3])[0]))

    writer = csv.writer(open('./data/valid_idx', 'wt'), delimiter='\t')
    for row in valid: 
        writer.writerow((row[0], row[1], json.loads(row[3])[0]))

    writer = csv.writer(open('./data/test_idx', 'wt'), delimiter='\t')
    for row in test:
        writer.writerow((row[0], row[1], json.loads(row[3])[0]))
else:
    os.system('rm -r ./data/')
    os.system('mkdir data/')
    PATH = os.path.join('../../heuristics/data','gathered_labels_'+DATASET+'.tsv')
    reader = csv.reader(open(PATH), delimiter='\t')
    writer = csv.writer(open('./data/test_idx','wt'),delimiter='\t')
    for row in reader:
        writer.writerow((row[0],row[1],json.loads(row[3])[0]))
