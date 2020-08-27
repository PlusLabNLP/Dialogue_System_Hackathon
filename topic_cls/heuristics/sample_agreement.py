import json
import pdb
import csv
from nltk import word_tokenize
data = json.load(open('train_comp.json','r'))
reader = csv.reader(open('./data/labels_with_kw.tsv','rt'),delimiter='\t')
lookup = { i[1] : i[2] for i in csv.reader(open('./entity_topic_assignment.csv', 'rt')) }
mapping = {'1':"fashion",'2':'politics','3':'books','4':"sports",'5':'genral entertainment','6':'music','7':'science & tech','8':'movie'}
#all_labels = {(row[0],row[1]):[row[2],json.loads(row[3]), None] for row in label}
#reader = csv.reader(open('./data/flat_data.tsv', 'rt'), delimiter='\t')
#flatten_data = { (row[0], row[1]) : row for row in reader  }
#keys = list(all_labels.keys())
#keys.remove(('conv_id','i'))
header = next(reader)
conv = {}
conv1 = []
conv2 = []
conv3 = []
for conv_id, i, rule, topic, agreement in reader:
    #print(agreement[2])
    #pdb.set_trace()
    if int(rule[-1]) in [2,3,4] and agreement[2]=='Y': 
        conv1.append((conv_id,int(i), json.loads(topic)))
        #pdb.set_trace()
        if conv_id not in conv:
            conv[conv_id] = [int(i)]
        else:
            conv[conv_id].append(int(i))
    elif int(rule[-1]) in [2,3,4] and agreement[2]=="N":
        conv2.append((conv_id,int(i),json.loads(topic)))
        #pdb.set_trace()
    elif int(rule[-1]) in [2,3,4] and agreement[2]=="G":
        conv3.append((conv_id, int(i), json.loads(topic)))
        #pdb.set_trace()

print(len(conv1)+len(conv2)+len(conv3))
print(len(conv1),len(conv2),len(conv3))
import random
printing = ["*********agreement accuracy estimation (sample 20)*********:", "*********agreement recall estimation (sample 20)*********", "*********general estimation (sample 20)*********"]
conv1sample = random.sample(conv1,20)
conv2sample = random.sample(conv2, 20)
conv3sample = random.sample(conv3, 20)
sample = [conv1sample, conv2sample, conv3sample]
for i in range(3):
    tmpSample = sample[i]
    print(printing[i])
    for idx, line, t in tmpSample:
        print(data[idx]['content'][line]['message'])
        try:
            print([mapping[k] for k in t])
        except:
            print([mapping[k] for k in t[0]])
print('-------------------------------------')
for idx, line, t in conv1+conv2+conv3:
    for k in t:
        try:
            mapping[k]
        except:
            try:
                [mapping[w] for w in t[0]]
            except:
                print(idx, line, t)
 
