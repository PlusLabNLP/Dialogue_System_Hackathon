from glob import glob
from convert import convert
import re
import os
import random 
import json
filelist = []

#files = glob('data/files/data/english/annotations/*.name')
def findfile(file_path):
	if os.path.isfile(file_path):
		if file_path.split('.')[1]=='name':
			filelist.append(file_path)
	else:
		for file_ls in os.listdir(file_path):
			findfile(os.path.join(file_path,file_ls))
	return filelist

findfile('data/files/data/english/annotations')
#print(filelist)
random.shuffle(filelist)

dataset = []
num = 0
for i in filelist:
	tmp, num = convert(i,'t.out',num)
	dataset.extend(tmp)

#random.shuffle(dataset)
print(len(dataset))

f1 = open('ontonote5/mrc-ner.train','w')
f2 = open('ontonote5/mrc-ner.dev','w')
f3 = open('ontonote5/mrc-ner.test','w')
json.dump(dataset[0:804870], f1, sort_keys=True, ensure_ascii=False, indent=2)
json.dump(dataset[804870:160974+804870], f2, sort_keys=True, ensure_ascii=False, indent=2)
json.dump(dataset[160974+804870:107316+160974+804870], f3, sort_keys=True, ensure_ascii=False, indent=2)
f1.close()
f2.close()
f3.close()
