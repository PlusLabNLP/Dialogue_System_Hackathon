import json
import csv
import pdb
from nltk import word_tokenize
data = json.load(open('train_comp.json','r'))
label = csv.reader(open('./data/labels_with_kw.tsv','rt'),delimiter='\t')
all_labels = {(row[0],row[1]):[row[2],json.loads(row[3]), json.loads(row[4]), None] for row in label}
reader = csv.reader(open('./data/flat_data.tsv', 'rt'), delimiter='\t')
flatten_data = { (row[0], row[1]) : row for row in reader  }
keys = list(all_labels.keys())
keys.remove(('conv_id','i'))
conv = {}

for conv_id, i in keys:
    if conv_id not in conv:
        conv[conv_id] = [int(i)]
    else:
        conv[conv_id].append(int(i))

 


for conv_id in data:
    for i in [0,1,-2,-1]:
        if data[conv_id]['content'][i]['keywords_2']==[] and data[conv_id]['content'][i]['topic']!='General':
            data[conv_id]['content'][i]['topic'] = ['General']
            data[conv_id]['content'][i]["entity_reading_set"] = ''
            try:
                all_labels[(conv_id,str(conv[conv_id][i]))][-1]=['9']
            except:
                print(conv_id,conv[conv_id][i])
                pdb.set_trace()

with open('train_comp_v2.json','w') as fw:
    json.dump(data,fw,sort_keys=False,ensure_ascii=False, indent=5)

writer = csv.writer(open('./data/labels_general.tsv','wt'),delimiter='\t')
for (conv_id,idx), [rulename, labels_, agreement,general] in all_labels.items():
    writer.writerow((conv_id, idx, rulename, json.dumps(general)))

#num1 = 0
#num2 = 0
#num3 = 0
#num4 = 0
#num5 = [0,0,0,0] # count General with FS
#num6 = [0,0,0,0] # count non-General with non-FS
#
#for conv_id in data:
#    if data[conv_id]['content'][0]['topic']==['General']:
#        num1+=1
#        if len(set(data[conv_id]['content'][0]['knowledge_source']) & set(['FS1','FS2','FS3']))>0:
#            num5[0]+=1
#    else:
#        if len(set(data[conv_id]['content'][0]['knowledge_source']) & set(['FS1','FS2','FS3']))==0:
#            num6[0]+=1
#
#    if data[conv_id]['content'][1]['topic']==['General']:
#        num2+=1
#        if len(set(data[conv_id]['content'][1]['knowledge_source']) & set(['FS1','FS2','FS3']))>0:
#            num5[1]+=1
#    else:
#        if len(set(data[conv_id]['content'][1]['knowledge_source']) & set(['FS1','FS2','FS3']))==0:
#            num6[1]+=1
#
#    if data[conv_id]['content'][-2]['topic']==['General']:
#        num3+=1
#        if len(set(data[conv_id]['content'][-2]['knowledge_source']) & set(['FS1','FS2','FS3']))>0:
#            num5[-2]+=1
#    else:
#        if len(set(data[conv_id]['content'][-2]['knowledge_source']) & set(['FS1','FS2','FS3']))==0:
#            num6[-2]+=1
#
#    if data[conv_id]['content'][-1]['topic']==['General']:
#        num4+=1 
#        if len(set(data[conv_id]['content'][-1]['knowledge_source']) & set(['FS1','FS2','FS3']))>0:
#            num5[-1]+=1
#    else:
#        if len(set(data[conv_id]['content'][-1]['knowledge_source']) & set(['FS1','FS2','FS3']))==0:
#            num6[-1]+=1
#print(len(data))
#print(num1+num2+num3+num4)
#print(num1, num2, num3, num4)
#print(num5)
#print(num6)
#
#
#


