import csv

general_labels = csv.reader(open('data/labels_general.tsv', 'rt'), delimiter='\t')
keyword_labels = csv.reader(open('data/labels_from_keywords.tsv', 'rt'), delimiter='\t')
rule_labels = csv.reader(open('data/labels_from_rules.tsv', 'rt'), delimiter='\t')
header = next(rule_labels)

gathered = csv.writer(open('data/gathered_labels.tsv', 'wt'), delimiter='\t')

for general_row, keyword_row, rule_row in zip(general_labels, keyword_labels, rule_labels):
    conv_id, idx = rule_row[0], rule_row[1]
    
    if general_row[3] != 'null':
        # general
        label = 'gold:general'
    if rule_row[2] == 'rule_1':
        # gold rule_1
        label = 'gold:rule_1'
    else:
        # double agreement
        if keyword_row[4] in rule_row[3]:
            label = 'gold:agreement:%s' % rule_row[2]
        else:
            label = rule_row[2]

    writerow = [ conv_id, idx, label ]
    gathered.writerow(writerow)

if next(general_labels, None) or next(keyword_labels, None) or next(rule_labels, None):
    raise Exception('Not all the label files were of the same length.')
