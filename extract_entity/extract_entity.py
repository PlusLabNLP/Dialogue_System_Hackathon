import json
from tqdm import tqdm

from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
model.save_pretrained('../saved/')
tokenizer.save_pretrained('../saved/')

#model = AutoModelForTokenClassification.from_pretrained('../saved/')
#tokenizer = AutoTokenizer.from_pretrained('../saved/')


label_list = [
    "O",       # Outside of a named entity
    "B-MISC",  # Beginning of a miscellaneous entity right after another miscellaneous entity
    "I-MISC",  # Miscellaneous entity
    "B-PER",   # Beginning of a person's name right after another person's name
    "I-PER",   # Person's name
    "B-ORG",   # Beginning of an organisation right after another organisation
    "I-ORG",   # Organisation
    "B-LOC",   # Beginning of a location right after another location
    "I-LOC"    # Location
]

with open('test_rare.json', 'r') as fin:
	data = json.load(fin)

# print(data)

all_keys = []
for key in tqdm(data):
	temp = {}
	tmp_url = data[key]["article_url"]
	tmp_config = data[key]["config"]
	tmp_content = []
	tmp_rating = data[key]["conversation_rating"]
	for msg in data[key]["content"]:
		# sequence = msg["message"].lower()
		sequence = msg["message"]

		# Bit of a hack to get the tokens with the special tokens
		tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence)))
		inputs = tokenizer.encode(sequence, return_tensors="pt")

		outputs = model(inputs)[0]
		predictions = torch.argmax(outputs, dim=2)

		# print([(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].tolist()) if label_list[prediction] != 'O'])
		entity = [(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].tolist())]

		# delete '##' before tokens
		r = []
		r_tags = []
		for i, tpl in enumerate(entity):
			if tpl[0].startswith("##"):
				if r:
					r[-1] += tpl[0][2:]
			else:
				r.append(tpl[0])
				r_tags.append(tpl[1])

		new_entity_token = [(i, j) for i, j in zip(r, r_tags)]
		# print(new_entity_token)

		# combine tokens into entities
		flag = False
		ent = []
		ent_tags = []
		for i, tpl in enumerate(new_entity_token):
			if tpl[1] == "O":
				flag = False
				continue
			elif tpl[1] == "I-MISC" or tpl[1] == "I-PER" or tpl[1] == "I-ORG" or tpl[1] == "I-LOC":
				if flag == False:
					flag = True
					ent.append(tpl[0])
					ent_tags.append(tpl[1])
				else:
					ent[-1] += ' '
					ent[-1] += tpl[0]
			elif tpl[1] == "B-MISC" or tpl[1] == "B-PER" or tpl[1] == "B-ORG" or tpl[1] == "B-LOC":
				ent.append(tpl[0])
				ent_tags.append(tpl[1])
				
		new_entity = [(i, j) for i, j in zip(ent, ent_tags)]
		# print(new_entity)

		# into json format
		tmp_dict = {"message" : msg["message"], 
		            "entity" : new_entity,
		            "agent" : msg["agent"], 
		            "sentiment" : msg["sentiment"], 
		            "knowledge_source" : msg["knowledge_source"], 
		            "turn_rating" : msg["turn_rating"]}
		tmp_content.append(tmp_dict)

	temp = {"article_url" : tmp_url, 
	        "config" : tmp_config,
	        "content" : tmp_content, 
	        "conversation_rating" : tmp_rating}
	all_keys.append({key:temp})

with open('output_1.json', "w") as f:
    json.dump(all_keys, f, sort_keys=False, ensure_ascii=False, indent=2)

