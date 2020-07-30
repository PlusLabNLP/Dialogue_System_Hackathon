import json
from tqdm import tqdm

with open('test_freq_out.json', 'r') as fin:
	data = json.load(fin)


ent_dict = {}
msg_cont = 0
ent_totl = 0
for item in tqdm(data):
	for key in item:
		for msg in item[key]["content"]:
			entity = msg["entity"]
			if entity:
				for ent in entity:
					if ent[1] not in ent_dict:
						ent_dict[ent[1]] = 1
						ent_totl += 1
					else:
						ent_dict[ent[1]] += 1
						ent_totl += 1
			msg_cont += 1

print(ent_totl, msg_cont, ent_totl/msg_cont)
print([(key, ent_dict[key]) for key in ent_dict])
print([(key, ent_dict[key]/msg_cont) for key in ent_dict])




