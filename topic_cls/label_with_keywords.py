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
conv_set = json.load(open('train_comp.json','r'))
reading_sets = json.load(open('../alexa-prize-topical-chat-dataset/reading_sets/post-build/train.json'))
vectors = Magnitude('./word2vec/GoogleNews-vectors-negative300.magnitude')
entity_keys = {}
for i in lookup:
    tmp = re.sub(r'[^\w\s]', '', i)
    tmp = re.sub(r'[0-9]+', '', tmp)
    tmp = word_tokenize(tmp.lower())
    entity_keys[i] = vectors.query(tmp)[-1]


#TODO
# rule2 <-> rule3 <-> rule5 merge
# rule4 <-> rule6 merge
# get rid of try/except in rules 2 and 3



def gold_label_classifier(conv_id,i,sentence, label, rule):
    
    if rule==1:
        return 'Y'

    #r = Rake()
    #r.extract_keywords_from_text(sentence)
    keywords = conv_set[conv_id]['content'][int(i)]["keywords_2"]

    if keywords==[]:
        return 'G'
    else:
        keyword1 = []
        keyword1 = [keyword1.extend(i.split()) for i in keywords]
        keyword1 = list(set(keyword1))
        #pdb.set_trace()
        #keyword1 = re.sub(r'[^\w\s]', '', keyword1)
        #keyword1 = re.sub(r'[0-9]+','',keyword1)
        keyword1_query = vectors.query(word_tokenize(keyword1.lower()))
        match_entity = ''
        dist = -1
        for key in entity_keys:
            dist_tmp = np.linalg.norm(np.matmul(keyword1_query,entity_keys[key].T))
            if dist_tmp>dist:
                dist = dist_tmp
                match_entity = key
        if lookup[match_entity] in label:
            return 'Y'
        else:
            return 'N'

rules = [ rule_1 , rule_2, rule_3, rule_4 ]
reader = csv.reader(open('./data/flat_data.tsv', 'rt'), delimiter='\t')
label = csv.reader(open('./data/labels.tsv','rt'),delimiter='\t')
data = { (row[0], row[1]) : row for row in reader  }
all_labels ={(row[0], row[1]):[row[2], row[3] ,None] for row in label}
label = csv.reader(open('./data/labels.tsv','rt'),delimiter='\t')
header = next(label)
for row in label:
    row = { i: j for i, j in zip(header, row)  }
    sentence = data[row['conv_id'],row['i']][3]
    topic = json.loads(all_labels[row['conv_id'], row['i']][1])
    i = int(row['none'][-1])
    #pdb.set_trace()
    agreement = gold_label_classifier(row['conv_id'],row['i'],sentence,topic,i)
    all_labels[row['conv_id'], row['i']][-1]=[agreement]


writer = csv.writer(open('./data/labels_with_kw.tsv', 'wt'), delimiter='\t')
for (conv_id, idx), [rulename, labels, agreement] in all_labels.items():
    writer.writerow((conv_id, idx, rulename, labels, json.dumps(agreement)))
