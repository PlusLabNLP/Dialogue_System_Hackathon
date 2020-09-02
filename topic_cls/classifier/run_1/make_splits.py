import csv
import json
import random
import sys
import os 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',type=str,help='which dataset is used in Alexa topical dataset for testing, options can be train, valid_rare, valida_freq, test_freq, test_rare',required=True,choices=['train','valid_rare','valid_freq','test_freq','test_rare'])
args = parser.parse_args()
DATASET = args.dataset # enter the dataset name to process, select one from train, test_freq, test_rare, valid_freq, valid_rare
PATH = os.path.join('../../heuristics/data','gathered_labels_'+DATASET+'.tsv')

# '/nas/home/zixiliu/new/Dialogue_System_Hackathon/topic_cls/heuristics/data/gathered_labels.tsv'
reader = csv.reader(open(PATH), delimiter='\t')


train, test = [], []
flag = True
for row in reader:
    if 'rule_1'  in row[2] :
        train.append(row)
        #continue
    elif 'agreement' in row[2]:
        test.append(row)
        #train.append(row)
    elif 'genera' in row[2]:
        test.append(row)
        #train.append(row)

random.shuffle(train)
valid_per = .3
valid_idx = int(len(train) * .3)

if DATASET=='train':
    os.system('rm -r ./data/')
    os.system('mkdir data/')
    writer = csv.writer(open('./data/train_idx', 'wt'), delimiter='\t')
    for row in train[valid_idx:]:
        writer.writerow((row[0], row[1], json.loads(row[3])[0]))

    writer = csv.writer(open('./data/valid_idx', 'wt'), delimiter='\t')
    for row in train[:valid_idx]:
        writer.writerow((row[0], row[1], json.loads(row[3])[0]))

    writer = csv.writer(open('./data/test_idx', 'wt'), delimiter='\t')
    for row in test:
        writer.writerow((row[0], row[1], json.loads(row[3])[0]))
else:
    os.system('rm -r ./data/')
    os.system('mkdir data/')
    reader = csv.reader(open(PATH), delimiter='\t')
    writer = csv.writer(open('./data/test_idx','wt'),delimiter='\t')
    for row in reader:
        writer.writerow((row[0],row[1],json.loads(row[3])[0]))
        

