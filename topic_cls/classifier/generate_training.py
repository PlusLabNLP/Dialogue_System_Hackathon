import csv
import json

reader = csv.reader(open('./data/flat_data.tsv', 'rt'), delimiter='\t')
header = next(reader)
data = { (row[0], int(row[1])) : row for row in reader }

reader = csv.reader(open('./data/labels.tsv', 'rt'), delimiter='\t')
writer = csv.writer(open('./data/examples.tsv', 'wt'), delimiter='\t')

for row in reader:
    # input
    utt = data[row[0], int(row[1])][3].strip()

    prev_utt = None
    if int(row[1]) != 0:
        prev_utt = data[row[0], int(row[1])-1][3].strip()

    if prev_utt:
        inp_str = '[CLS] %s [SEP] %s [SEP]' % (prev_utt, utt)
    else:
        inp_str = '[CLS] [SEP] %s [SEP]' % (utt)

    # output
    # just use the first label for now
    labels = json.loads(row[3])
    if len(labels) == 0:
        continue
    else:
        label = labels[0]

    writer.writerow([inp_str, label])
