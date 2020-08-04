import csv
import json

reader = csv.reader(open('./data/flat_data.tsv', 'rt'), delimiter='\t')
header = next(reader)
data = { (row[0], int(row[1])) : row for row in reader }

reader = csv.reader(open('./data/labels.tsv', 'rt'), delimiter='\t')
writer = csv.writer(open('./examples.tsv', 'wt'), delimiter='\t')

for row in reader:
    utt = data[row[0], int(row[1])][3].strip()

    prev_utt = None
    if int(row[1]) != 0:
        prev_utt = data[row[0], int(row[1])-1][3].strip()

    label
