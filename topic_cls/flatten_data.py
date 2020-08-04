import csv
import json
import pdb
import pickle

conversations = json.load(open('../alexa-prize-topical-chat-dataset/conversations/train.json'))
reading_sets = json.load(open('../alexa-prize-topical-chat-dataset/reading_sets/post-build/train.json'))
writer = csv.writer(open('./data/flat_data.tsv', 'wt', newline='\n'), delimiter='\t')

header = ['conv_id', 'i', 'agent', 'utt', 'knowledge_sources', 'knowledge_strs', 'entities' ]
writer.writerow(header)

for conv_id, conversation in conversations.items():
    for i, turn in enumerate(conversation['content']):
        agent = turn['agent']
        utt = turn['message']

        # get knowledge strings
        knowledge_base = reading_sets[conv_id]
        knowledge_sources = turn['knowledge_source']

        knowledge_strs = []
        entities = []
        for source in knowledge_sources:
            if source == 'Personal Knowledge':
                pass
            if source.startswith('FS'):
                k = knowledge_base[agent][source]['fun_facts'][int(source[-1])]
                e = knowledge_base[agent][source]['entity']
            if source.startswith('AS'):
                k = knowledge_base['article'][source]
                e = [ r['entity'] for r in knowledge_base[agent].values() ]
            entities.append(e)
            knowledge_strs.append(k)

        row = [ conv_id, i, agent, utt, json.dumps(knowledge_sources), json.dumps(knowledge_strs), json.dumps(entities) ]
        writer.writerow(row)