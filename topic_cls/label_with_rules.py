import csv
import json
import pickle
import codecs
import collections

lookup = { i[1] : i[2] for i in csv.reader(open('./entity_topic_assignment.csv', 'rt')) }

def rule_1(data, row, all_labels):
    labels = []
    for k, e in zip(json.loads(row['knowledge_sources']), json.loads(row['entities'])):
        if k.startswith('FS'):
            labels.append(lookup[e])

    return list(set(labels))

def rule_2(data,row, all_labels):
    try:
        k1 = int(row['i'])-1
        rulename, prev_topic = all_labels[(row['conv_id'],k1)]
    except:
        prev_topic = None
    try:
        k2 = int(row['i'])+1
        rulename,next_topic = all_labels[(row['conv_id'], k2)]
    except:
        next_topic =None
    if not (prev_topic and next_topic):
        return []
    elif not prev_topic and next_topic:
        return next_topic
    elif prev_topic and not next_topic:
        return prev_topic
    else:
        return(list(set(prev_topic) & set(next_topic)))
       
#def rule_3(data,row,)



   

rules = [ rule_1 , rule_2]

reader = csv.reader(open('./data/flat_data.tsv', 'rt'), delimiter='\t')
data = { (row[0], row[1]) : row for row in reader }
all_labels = {i:('none',None) for i in data.keys()}
for i, rule in enumerate(rules,1):
    reader = csv.reader(open('./data/flat_data.tsv', 'rt'), delimiter='\t')
    header = next(reader)
    for row in reader:
        row = { i: j for i, j in zip(header, row) }
        rule_name = 'rule_%d' % i
        labels = rule(data, row,all_labels)
        if (len(labels))>0:
            all_labels[(row['conv_id'],row['i'])] = (rule_name ,labels)

writer = csv.writer(open('./data/labels.tsv', 'wt'), delimiter='\t')
for (conv_id, idx), (rulename, labels) in all_labels.items():
    #row = { i: j for i, j in zip(header, row)  }
    writer.writerow((conv_id, idx, rulename, json.dumps(labels)))
