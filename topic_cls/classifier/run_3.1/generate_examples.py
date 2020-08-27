import csv
import json
import argparse
import os

def generate_examples(idx, out, src):
    #reader = csv.reader(open('/nas/home/zixiliu/new/Dialogue_System_Hackathon/topic_cls/heuristics/data/flat_data.tsv', 'rt'), delimiter='\t')
    reader = csv.reader(open('../../heuristics/data/flat_data_'+src+'.tsv', 'rt'), delimiter='\t')
    header = next(reader)
    data = { (row[0], int(row[1])) : row for row in reader }

    reader = csv.reader(open(idx, 'rt'), delimiter='\t')
    writer = csv.writer(open(out, 'wt'))
    writer.writerow(('text', 'label'))

    for row in reader:
        # input
        utt = data[row[0], int(row[1])][3].strip()

        prev_utt = None
        if int(row[1]) != 0:
            prev_utt = data[row[0], int(row[1])-1][3].strip()

        if prev_utt:
            inp_str = '[CLS] %s [SEP] %s [SEP]' % (prev_utt, utt)
        else:
            inp_str = '[CLS] [SEP] %s [SEP]' % (utt)

        # output
        # just use the first label for now
        labels = row[2]
        if len(labels) == 0:
            continue
        else:
            label = labels[0]

        writer.writerow([inp_str, label])

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',type=str,help='which dataset is used in Alexa topical dataset for testing, options can be train, valid_rare, valida_freq, test_freq, test_rare',required=True,choices=['train','valid_rare','valid_freq','test_freq','test_rare'])
args = parser.parse_args()
DATASET = args.dataset # select from train, valid_rare, valid_freq, test_freq, test_rare

if DATASET=='train':
    generate_examples('data/train_idx', 'data/train.csv',DATASET)
    generate_examples('data/valid_idx', 'data/valid.csv',DATASET)
else:
    generate_examples('data/test_idx', 'data/test.csv',DATASET)
