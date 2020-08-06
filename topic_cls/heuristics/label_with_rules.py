import csv
import re
import json
import pickle
import codecs
import collections
import pdb
from collections import Counter
from rake_nltk import Rake
from pymagnitude import *
from nltk.tokenize import word_tokenize
import numpy as np
lookup = { i[1] : i[2] for i in csv.reader(open('./entity_topic_assignment.csv', 'rt')) }

reading_sets = json.load(open('../alexa-prize-topical-chat-dataset/reading_sets/post-build/train.json'))
#TODO
# rule2 <-> rule3 <-> rule5 merge
# rule4 <-> rule6 merge
# get rid of try/except in rules 2 and 3

def rule_1(data, row, all_labels,most_common_topic=None):
    labels = []
    for k, e in zip(json.loads(row['knowledge_sources']), json.loads(row['entities'])):
        if k.startswith('FS'):
            labels.append(lookup[e])

    return list(set(labels))

def rule_2( data,row, all_labels,most_common_topic=None ):
    try:
        k1 = str(int(row['i'])-1)
        rulename1, prev_topic = all_labels[(row['conv_id'],k1)]

        if rulename1 != 'rule_1':
            raise Exception
    except:
        rulename1 = 'none'
        prev_topic = None
    try:
        k2 = str(int(row['i'])+1)
        rulename2,next_topic = all_labels[(row['conv_id'], k2)]

        if rulename2 != 'rule_1':
            raise Exception
    except:
        rulename2 = 'none'
        next_topic =None

    if not (prev_topic or next_topic):
        return []
    elif not prev_topic and next_topic:
            return next_topic
    elif prev_topic and not next_topic:
            return prev_topic
    else:
        return list(set(prev_topic) & set(next_topic))
       
def rule_3(data, row, all_labels,most_common_topic=None  ):
    try:
        k1 = str(int(row['i'])-1)
        rulename1, prev_topic = all_labels[(row['conv_id'],k1)]

        if rulename1 != 'rule_1':
            raise Exception
    except:
        rulename1 = 'none'
        prev_topic = [] 
    try:
        k2 = str(int(row['i'])+1)
        rulename2,next_topic = all_labels[(row['conv_id'], k2)]

        if rulename2 != 'rule_1':
            raise Exception
    except:
        rulename2 = 'none'
        next_topic = []

    return list(set(prev_topic) | set(next_topic))


def rule_4(data, row, all_labels,most_common_topic=None  ):
    k1 = str(int(row['i'])-1)
    rulename1, topic1 = all_labels.get((row['conv_id'], k1), ('none', None))

    k2 = str(int(row['i']))
    rulename2, topic2 = all_labels.get((row['conv_id'], k2), ('none', None))

    k3 = str(int(row['i'])+1)
    rulename3, topic3 = all_labels.get((row['conv_id'], k2), ('none', None))

    labels = []
    if rulename2 == 'none' and rulename1 in ['rule_2','rule_3', 'none','rule_4'] and rulename3 in ['rule_2','rule_3', 'none','rule_4']:
        labels = [most_common_topic ]
    return labels
 

rules = [ rule_1 , rule_2, rule_3, rule_4 ]

reader = csv.reader(open('./data/flat_data.tsv', 'rt'), delimiter='\t')
data = { (row[0], row[1]) : row for row in reader }
all_labels = {i:('none',None) for i in data.keys()}
for i, rule in enumerate(rules,1):
    reader = csv.reader(open('./data/flat_data.tsv', 'rt'), delimiter='\t')
    header = next(reader)

    rule_name = 'rule_%d' % i
    for row in reader:
        row = { i: j for i, j in zip(header, row) }
        # TODO
        # most common topic needs to be refactored
        # into the fourth rule, putting it here is confusing
        topics = []
        if row['i']=='0' and  i==4:
            counter = Counter()
            for j in range(0, 55):
                try:
                    rulename, topic = all_labels[row['conv_id'], str(j)]

                    topics.append(topic)
                except:
                    pass
            counter = Counter(k[0] for k in topics if k is not None)
            try:
                most_common_topic = counter.most_common(1)[0][0]
            except:
                t = [lookup[reading_sets[row['conv_id']]['agent_1'][fs]['entity']] for fs in ['FS1','FS2', 'FS3']]
                most_common_topic = t

        if all_labels[row['conv_id'], row['i']] != ('none', None):
            continue

        labels = rule( data, row,all_labels,most_common_topic) if i==4 else rule( data, row,all_labels,None)
        if (len(labels))>0:
            all_labels[(row['conv_id'],row['i'])] = (rule_name ,labels)
writer = csv.writer(open('./data/labels.tsv', 'wt'), delimiter='\t')
for (conv_id, idx), (rulename, labels) in all_labels.items():
    writer.writerow((conv_id, idx, rulename, json.dumps(labels)))
