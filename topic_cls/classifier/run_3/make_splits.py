import csv
import json
import random

reader = csv.reader(open('/nas/home/zixiliu/new/Dialogue_System_Hackathon/topic_cls/heuristics/data/gathered_labels.tsv'), delimiter='\t')

train, test, general = [], [], []
flag = True
for row in reader:
    if 'rule_1' in row[2]:
        train.append(row)
    elif 'agreement' in row[2]:
        test.append(row)
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
writer = csv.writer(open('./data/train_idx', 'wt'), delimiter='\t')
for row in train[valid_idx:]:
    writer.writerow((row[0], row[1], json.loads(row[3])[0]))

writer = csv.writer(open('./data/valid_idx', 'wt'), delimiter='\t')
for row in train[:valid_idx]:
    writer.writerow((row[0], row[1], json.loads(row[3])[0]))

writer = csv.writer(open('./data/test_idx', 'wt'), delimiter='\t')
for row in test:
    writer.writerow((row[0], row[1], json.loads(row[3])[0]))
