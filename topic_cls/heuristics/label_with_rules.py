import csv
import re
import json
import pickle
import codecs
import collections
import pdb
from collections import Counter
#from rake_nltk import Rake
#from pymagnitude import *
#from nltk.tokenize import word_tokenize
#import numpy as np
import sys
import os
lookup = { i[1] : i[2] for i in csv.reader(open('./entity_topic_assignment.csv', 'rt')) }
src = sys.argv[1]# input from 'train','test_freq','test_rare','valid_freq','valid_rare'
reading_sets = json.load(open(os.path.join('../../alexa-prize-topical-chat-dataset/reading_sets/post-build',src+'.json')))
#TODO
# rule2 <-> rule3 <-> rule5 merge
# rule4 <-> rule6 merge
# get rid of try/except in rules 2 and 3

def rule_1(data, row, all_labels,most_common_topic=None):
    labels = []
    for k, e in zip(json.loads(row['knowledge_sources']), json.loads(row['entities'])):
        if k.startswith('FS'):
            labels.append(lookup[e])
    labels = sorted(list(set(labels)))
    return labels

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
        # update entities in flat data
        data[row['conv_id'], row['i']][4] = data[row['conv_id'],str(int(row['i'])+1)][4]
        return next_topic
    elif prev_topic and not next_topic:
        # update entities in flat data
        data[row['conv_id'],row['i']][4] = data[row['conv_id'],str(int(row['i'])-1)][4]
        return prev_topic
    else:
        #update entities in flat data
        prev_entities = data[row['conv_id'],str(int(row['i'])-1)][4]
        next_entities = data[row['conv_id'],str(int(row['i'])+1)][4]
        data[row['conv_id'],row['i']][4] = sorted(list(set(prev_entities)&set(next_entities)))
        if data[row['conv_id'],row['i']][4]==[]:
            data[row['conv_id'],row['i']][4] = sorted(list(set(prev_entities)|set(next_entities)))
        return sorted(list(set(prev_topic) & set(next_topic)))
       
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
    # update entities in data
    if prev_topic!=[] and next_topic!=[]:
        prev_entities = data[row['conv_id'],str(int(row['i'])-1)][4]
        next_entities = data[row['conv_id'],str(int(row['i'])+1)][4]
        data[row['conv_id'],row['i']][4] = sorted(list(set(prev_entities)|set(next_entities)))
    return sorted(list(set(prev_topic) | set(next_topic)))


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
    # since most common topic is not reliable, flat data is not updated in rule_4
    return labels
 

rules = [ rule_1 , rule_2, rule_3, rule_4 ]

reader = csv.reader(open('./data/flat_data_'+src+'.tsv', 'rt'), delimiter='\t')
#data = { (row[0], row[1]) : row for row in reader }
header = next(reader)
data = {}
for row in reader:
    data [row[0], row[1]]= [row[2],row[3], json.loads(row[4]), json.loads(row[5]), json.loads(row[6]) ]
all_labels = {i:('none',None) for i in data.keys()}
for i, rule in enumerate(rules,1):
    reader = csv.reader(open('./data/flat_data_'+src+'.tsv', 'rt'), delimiter='\t')
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

        if len(labels)>0:
            all_labels[(row['conv_id'],row['i'])] = (rule_name ,labels)
writer = csv.writer(open('./data/labels_from_rules_'+src+'.tsv', 'wt'), delimiter='\t')
writer.writerow(('conv_id','i','none','null'))
for (conv_id, idx), (rulename, labels) in all_labels.items():
    writer.writerow((conv_id, idx, rulename, json.dumps(labels)))
writer = csv.writer(open('./data/flat_data_'+src+'_update entities.tsv', 'wt', newline='\n'), delimiter='\t')
header = ['conv_id', 'i', 'agent', 'utt', 'knowledge_sources', 'knowledge_strs', 'entities' ]
writer.writerow(header)
for (conv_id, i), (agent, utt, knowledge_sources, knowledge_strs,entities) in data.items():
    row = [conv_id, i, agent, utt, json.dumps(knowledge_sources), json.dumps(knowledge_strs), json.dumps(entities)]
    writer.writerow(row)
