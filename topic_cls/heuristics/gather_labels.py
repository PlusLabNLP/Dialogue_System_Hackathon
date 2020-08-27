import csv
import json
import sys
import os

src = sys.argv[1]# input from 'train','test_freq','test_rare','valid_freq','valid_rare'
general_labels = csv.reader(open('data/labels_general_'+src+'.tsv', 'rt'), delimiter='\t')
keyword_labels = csv.reader(open('data/labels_from_keywords_'+src+'.tsv', 'rt'), delimiter='\t')
rule_labels = csv.reader(open('data/labels_from_rules_'+src+'.tsv', 'rt'), delimiter='\t')
header = next(rule_labels)

gathered = csv.writer(open('data/gathered_labels'+'_'+src+'.tsv', 'wt'), delimiter='\t')

for general_row, keyword_row, rule_row in zip(general_labels, keyword_labels, rule_labels):
    conv_id, idx = rule_row[0], rule_row[1]
    
    if general_row[3] != 'null':
        # general
        label = 'gold:general'
        print('here')
        topics = [ '9' ]
    elif rule_row[2] == 'rule_1':
        # gold rule_1
        label = 'gold:rule_1'
        topics = json.loads(rule_row[3])
    else:
        # double agreement
        if keyword_row[4] in rule_row[3]:
            label = 'gold:agreement:%s' % rule_row[2]
            topics = [ keyword_row[4] ]
        else:
            label = rule_row[2]
            topics = json.loads(rule_row[3])

    writerow = [ conv_id, idx, label, json.dumps(topics) ]
    gathered.writerow(writerow)

if next(general_labels, None) or next(keyword_labels, None) or next(rule_labels, None):
    raise Exception('Not all the label files were of the same length.')
