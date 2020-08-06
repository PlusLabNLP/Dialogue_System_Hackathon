import csv
import re
import json
import pickle
import codecs
import collections
import pdb
from collections import Counter
from rake_nltk import Rake
from pymagnitude import *
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import numpy as np

lookup = { i[1] : i[2] for i in csv.reader(open('./entity_topic_assignment.csv', 'rt')) }

# read in entity vectors
vectors = Magnitude('./GoogleNews-vectors-negative300.magnitude')

known_entities = []
entities_to_str = {}
for i in lookup.keys():
    tmp = re.sub(r'[^\w\s]', '', i)
    tmp = re.sub(r'[0-9]+', '', tmp)
    tmp = word_tokenize(tmp.lower())
    entities_to_str[tmp[-1]] = i
    known_entities.append(tmp[-1])

def nearest_neighbor_labelling(sentence):
    r = Rake(min_length=2, max_length=4)
    r.extract_keywords_from_text(sentence)
    keywords = r.get_ranked_phrases()

    if keywords == []:
        return ('none', 'none', '9')
    else:
        first_keyword = keywords[0]
        last_word = word_tokenize(first_keyword)[-1]
        matched_entity = vectors.most_similar_to_given(last_word, known_entities)

    return first_keyword, entities_to_str[matched_entity], lookup[entities_to_str[matched_entity]]
        
reader = csv.reader(open('./data/flat_data.tsv', 'rt'), delimiter='\t')
header = next(reader)

total = len(list(open('./data/flat_data.tsv', 'rt')))
writer = csv.writer(open('./data/labels_from_keywords.tsv', 'wt'), delimiter='\t')

for row in tqdm(reader, total=total):
    row = { i: j for i, j in zip(header, row) }

    if 'FS' in row['knowledge_sources']:
        # don't label rule 1
        first_keyword, matched_entity, topic = ('none', 'none', 'none')
    else:
        first_keyword, matched_entity, topic = nearest_neighbor_labelling(row['utt'])

    writer.writerow((row['conv_id'], row['i'], first_keyword, matched_entity, topic))
