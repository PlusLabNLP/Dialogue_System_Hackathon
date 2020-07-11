import os
import re
import json

def convert(filename, dst, num):
	query = {
		"LAW": "law entities are named documents made into laws",
		"EVENT": "event entities are named hurricanes, battles, wars, sports events",
		"CARDINAL": "cardinal entities are numerals that do not fall under another type",
		"FAC": "facility entities are buildings, airports, highways, bridges, etc",
		"TIME": "time entities are times smaller than a day",
		"DATE": "data entities are absolute or relative dates or periods",
		"ORDINAL": "ordinal entities are first or second",
		"ORG": "organization entities are companies, agencies, institutions",
		"QUANTITY": "quantity entities are measurements, as of weight or distance",
		"PERCENT": "percent entities are percentage including %",
		"WORK_OF_ART": "work of art entities are titles of books, songs",
		"LOC": "location entities are limited to not geographical locations, mountain ranges, bodies of water",
		"LANGUAGE": "language entities are limited to any named language",
		"NORP": "nationalities or religious or political groups",
		"MONEY": "money entities are limited to monetary values, including unit",
		"PERSON": "a person entity is limited to people, including fictional",
		"GPE": "geographical political entities are countries, cities, state",
		"PRODUCT": "product entities are limited to vehicles, weapons, foods, etc. not services"
	}


	onto_ner_dataset = []


	with open(filename, 'r') as fin:
		lines = fin.readlines()
		#lines = list(filter(lambda x:x!='%pw\n',lines))
		#lines = list(filter(lambda x:x!='',lines))
		#lines.remove('%pw\n')
		#rgx_all = []
		count = 0
		tmp_qas_1 = num
		for line in lines:
			# print(count, line)
			#if line[-1] not in ['.','!',',','?',')','(',';',':']:
			line = line.strip()
			#line = re.sub(r'\s%pw','',line)
			#line = re.sub(r'%pw\s','',line)
			if count == 0 or count == len(lines)-1 or line=='' or 'ENAMEX' not in line:
				count += 1
				continue

			text1 = line
			ner = {}
			WRONG_ANNOTATION_FLAG = False
			while text1.find('ENAMEX')>0:
				tmp = re.search(r'<ENAMEX\s+TYPE=\"(.+?)\">(.+?)</ENAMEX>',text1)
				span = tmp.span()
				tmp_split = text1[0:span[0]]
				span_tr = (len(tmp_split.split()),len(tmp_split.split())+len(tmp.group(2).split()))
				#print(span_tr)
				text1 = text1[:span[0]]+tmp.group(2)+text1[span[1]:]
				#print(text1)
				temp = text1.split()
				try:
					assert(' '.join(temp[span_tr[0]:span_tr[1]])==tmp.group(2))
				except:
					print(' '.join(temp[span_tr[0]:span_tr[1]]))
					print(tmp.group(2))
					print(filename)
					print(text1)
					print(line)
					WRONG_ANNOTATION_FLAG = True
					break
					#assert(' '.join(temp[span_tr[0]:span_tr[1]])==tmp.group(2))
				if tmp.group(1) in ner:
					ner[tmp.group(1)].append(span_tr)
				else:
					ner[tmp.group(1)] = [span_tr]
			if WRONG_ANNOTATION_FLAG==True:
				count += 1
				continue

			try:
				if text1[-1] not in ['.','!',',','?',')','(',';',':','<','>']:
					text1 = text1.strip()+' .'
			except:
				print(len(text1))
				print(text1)
				text1[-1]

			count += 1
			#print(rgx_all)
			# [('PERSON', 'Al Gore'), ('DATE', 'Thanksgiving'), ('GPE', 'Washington')]
			#print(tmp_sent)

			
			# tmp_start = []
			# tmp_end = []
			# tmp_span = []
			tmp_qas_2 = 1

			for tmp_lbl in query:
				tmp_start = []
				tmp_end = []
				tmp_span = []
				tmp_qry = query[tmp_lbl]
				tmp_imp = True
				


				if tmp_lbl in ner:
					tmp_imp = False
					for i in ner[tmp_lbl]:
						tmp_start.append(i[0])
						tmp_end.append(i[1]-1)
						tmp_span.append(str(i[0])+';'+str(i[1]-1))
				onto_ner_dataset.append({
					"context": text1,
					"end_position": tmp_end,
					"entity_label": tmp_lbl,
					"impossible": tmp_imp,
					"qas_id": "{}.{}".format(str(tmp_qas_1), str(tmp_qas_2)),
					"query": tmp_qry,
					"span_position": tmp_span,
					"start_position": tmp_start
				})
				tmp_qas_2 += 1
			tmp_qas_1 += 1
			# print(onto_ner_dataset)

	#with open(dst, "a") as f:
	#	json.dump(onto_ner_dataset, f, sort_keys=True, ensure_ascii=False, indent=2)
	return onto_ner_dataset, tmp_qas_1





