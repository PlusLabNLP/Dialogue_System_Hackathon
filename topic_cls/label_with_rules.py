import csv
import json
import pickle
import codecs
import collections

lookup = { i[1] : i[2] for i in csv.reader(open('./entity_topic_assignment.csv', 'rt')) }

def rule_1(data, row):
    labels = []
    for k, e in zip(json.loads(row['knowledge_sources']), json.loads(row['entities'])):
        if k.startswith('FS'):
            labels.append(lookup[e])

    return list(set(labels))

rules = [ rule_1 ]

reader = csv.reader(open('./data/flat_data.tsv', 'rt'), delimiter='\t')
data = { (row[0], row[1]) : row for row in reader }

writer = csv.writer(open('./data/labels.tsv', 'wt'), delimiter='\t')
reader = csv.reader(open('./data/flat_data.tsv', 'rt'), delimiter='\t')
header = next(reader)
for row in reader:
    row = { i: j for i, j in zip(header, row) }

    for i, rule in enumerate(rules):
        rule_name = 'rule_%d' % i
        labels = rule(data, row)
        
        if len(labels) > 0:
            break

    if len(labels) == 0:
        rule_name = 'none'
        labels = []

    writer.writerow((row['conv_id'], row['i'], rule_name, json.dumps(labels)))
