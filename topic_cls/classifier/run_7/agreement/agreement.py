import csv
import json

reader = csv.reader(open('labels_from_keywords_train.tsv','rt'),delimiter='\t')
all_data = {}
for row in reader:
    all_data[(row[0],row[1])] = row[4]

reader = csv.reader(open('merge_heuristics','rt'),delimiter='\t')
agree12 = []
agree13 = []
agree23 = []
agree123 = []
general = []
for row in reader:
    tmp = json.loads(row[3])
    if 'gold' in row[2] and 'agreement' in row[2]:
        if row[4] in tmp:
            agree123.append([row[0],row[1],row[4]])
            #print(row[2], tmp, row[4])
            #break
        else:
            agree12.append([row[0],row[1],row[4]])
            #print(row[2], tmp, row[4])
            #break
    elif 'gold' not in row[2]:
        if row[4] in tmp:
            agree13.append([row[0],row[1],row[4]])
            #print(row[2], tmp, row[4])
            #break
        else:
            if all_data[(row[0],row[1])]==row[4]:
                agree23.append([row[0],row[1],row[4]])
                #print(row[2], tmp, row[4])
                #print(all_data[(row[0],row[1])],row[4])
                #break
    elif 'general' in row[2]:
        if row[4] in tmp:
            general.append([row[0],row[1],row[4]])

print('agree12 ',len(agree12))
print('agree13 ',len(agree13))
print('agree23 ',len(agree23))
print('agree123 ',len(agree123))
print('general',len(general))
#for idx,agree in zip(['12','13','23','123'],[agree12, agree13, agree23,agree123]):
#    writer = csv.writer(open('heuristic_'+idx+'.tsv','wt'), delimiter='\t')
#    for conv_id, i, label in agree:
#        writer.writerow((conv_id,i,label))
