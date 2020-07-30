import json

from flair.data import Sentence
from flair.models import SequenceTagger

# # make a sentence
# sentence = Sentence("I'd rather watch it inside where it's warm.  Have you heard about the Georgia Tech-Cumberland game of 1916?")

# # load the NER tagger
# tagger = SequenceTagger.load('ner')

# # run NER over sentence
# tagger.predict(sentence)

# print(sentence)
# print('The following NER tags are found:')

# # iterate over entities and print
# for entity in sentence.get_spans('ner'):
#     print(entity)


with open('test_freq_1.json', 'r') as fin:
	data = json.load(fin)

# print(data)

all_keys = []
for key in data:
	temp = {}
	tmp_url = data[key]["article_url"]
	tmp_config = data[key]["config"]
	tmp_content = []
	tmp_rating = data[key]["conversation_rating"]
	for msg in data[key]["content"]:

		message = Sentence(msg["message"])
		tagger = SequenceTagger.load('ner')
		tagger.predict(message)
		tmp_entity = []
		for entity in message.get_spans('ner'):
			print(str(entity.__dict__), type(entity.__dict__))
			tmp_entity.append(str(entity.__dict__))
		tmp_dict = {"message" : msg["message"], 
		            "entity" : tmp_entity,
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

with open('output1.json', "w") as f:
    json.dump(all_keys, f, sort_keys=False, ensure_ascii=False, indent=2)

