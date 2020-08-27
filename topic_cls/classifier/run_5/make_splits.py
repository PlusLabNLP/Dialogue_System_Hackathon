import csv
import json
import random
import os 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',type=str,help='which dataset is used in Alexa topical dataset for testing, options can be train, valid_rare, valida_freq, test_freq, test_rare',required=True,choices=['train','valid_rare','valid_freq','test_freq','test_rare'])
args = parser.parse_args()

DATASET = args.dataset # enter the dataset name to process, select one from train, test_freq, test_rare, valid_freq, valid_rare


if DATASET=='train':
    reader = csv.reader(open(os.path.join('../../heuristics/data','gathered_labels_'+DATASET+'.tsv')), delimiter='\t')
    for row in reader:
        if 'rule_1' in row[2]:
            train.append(row)
        elif 'agreement' in row[2]: test.append(row)
        elif 'general' in row[2]:
            general.append(row)

    random.shuffle(general)
    general1 = general[:int(len(general)*(len(train)/(len(train)+len(test))))]
    general2 = general[int(len(general)*(len(train)/(len(train)+len(test)))):]
    train = train+general1
    test = test + general2

    random.shuffle(train)
    valid_per = .3
    valid_idx = int(len(train) * .3)
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
    PATH = os.path.join('../../heuristics/data','gathered_labels_'+DATASET+'.tsv')
    reader = csv.reader(open(PATH), delimiter='\t')
    writer = csv.writer(open('./data/test_idx','wt'),delimiter='\t')
    for row in reader:
        writer.writerow((row[0],row[1],json.loads(row[3])[0]))
 
