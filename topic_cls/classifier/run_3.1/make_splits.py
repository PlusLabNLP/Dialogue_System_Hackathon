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


if DATASET=='train':
    reader = list(csv.reader(open('data_run_3/test_idx','rt'),delimiter='\t'))
    train, test = [], []
    flag = True
    random.shuffle(reader)
    train = reader[:int(len(reader)*0.5)]
    test = reader[int(len(reader)*0.5):]
    writer = csv.writer(open('./data/train_idx', 'wt'), delimiter='\t')
    for row in train:
        writer.writerow((row[0], row[1], row[2]))

    writer = csv.writer(open('./data/valid_idx', 'wt'), delimiter='\t')
    for row in test:
        writer.writerow((row[0], row[1], row[2]))
else:
    os.system('rm -r ./data/')
    os.system('mkdir data/')
    PATH = os.path.join('../../heuristics/data','gathered_labels_'+DATASET+'.tsv')
    reader = csv.reader(open(PATH), delimiter='\t')
    writer = csv.writer(open('./data/test_idx','wt'),delimiter='\t')
    for row in reader:
        writer.writerow((row[0],row[1],json.loads(row[3])[0]))
 
