import json
from data_labeling.data_labeling import label
import os
import yake
from tqdm import tqdm
import argparse
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch
import csv

n_gram=3

def extract_entities(data, device):

    num_convs = len(list(data.keys()))
    print('Extracting entities for all utterances of {} conversations ...'.format(num_convs))
    
    model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
    model.to(device)
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

    all_keys = {}
    
    for key in tqdm(data):
        temp = {}
        tmp_url = data[key]["article_url"]
        tmp_config = data[key]["config"]
        tmp_content = []
        tmp_rating = data[key]["conversation_rating"]
        for msg in data[key]["content"]:
            sequence = msg["message"]

            # Bit of a hack to get the tokens with the special tokens
            tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence)))
            inputs = tokenizer.encode(sequence, return_tensors="pt").to(device)
            
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
                        "keywords_1": msg["keywords_1"],
                        "keywords_2": msg["keywords_2"],
                        "keywords_3": msg["keywords_3"],
                        "agent" : msg["agent"], 
                        "sentiment" : msg["sentiment"], 
                        "knowledge_source" : msg["knowledge_source"], 
                        "turn_rating" : msg["turn_rating"]}
            tmp_content.append(tmp_dict)

        temp = {"article_url" : tmp_url, 
                "config" : tmp_config,
                "content" : tmp_content, 
                "conversation_rating" : tmp_rating}
        all_keys[key] = temp
    print('Done!')
    
    return all_keys

def entities_stats(data):
    ent_dict = {}
    msg_cont = 0
    ent_totl = 0
    for key in tqdm(data):
        for msg in data[key]["content"]:
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

def sortFunc(e):
    return len(e.split())

#configuration1:extract trigram, bigram and unigram without overlapping
def get_kwd_1(keywords):
    outputs = []
    keys = [keyword for score, keyword in keywords]
    sorted_keywords = sorted(keys, key=sortFunc, reverse=True)
    for keyword in sorted_keywords:
        terms = keyword.split()
        for i in range(n_gram, 1, -1):
            is_in = [True for k in outputs if keyword in k]
            if True not in is_in:
                outputs.append(keyword)    
    return outputs

#configuration2: get the keywords with best scores
def get_kwd_2(keywords):
    keys = [keyword for score, keyword in keywords]
    return keys[:3]

#configuration3: get the unigram keywords only
def get_kwd_3(keywords):
    outputs = []
    keys = [keyword for score, keyword in keywords]
    for key in keys:
        if len(key.split()) == 1:
            outputs.append(key)

    return outputs

def extract_keywords(data):
    special_inputs =[':)', '.']
    kw_extractor = yake.KeywordExtractor(n=n_gram)
    num_convs = len(list(data.keys()))
    print('Extracting keywords for all utterances of {} conversations ...'.format(num_convs))
    new_data = {}
    new_conv = {}
    
    for c in tqdm(range(num_convs)):
        conv = list(data.values())[c]['content']
        new_conv= list(data.values())[c]
        num_utts = len(conv)
        for u in range(num_utts):
            utterance = conv[u]['message']
            if utterance in special_inputs:
                utterance='i'
            kws = kw_extractor.extract_keywords(utterance)
            #print(kws)
            kws_1 = get_kwd_1(kws)
            kws_2 = get_kwd_2(kws)
            kws_3 = get_kwd_3(kws)
            #print('kw1: {}'.format(kws_1))
            #print('kw2: {}'.format(kws_2))
            #print('kw3: {}'.format(kws_3))
            new_conv['content'][u]['keywords_1'] = kws_1
            new_conv['content'][u]['keywords_2'] = kws_2
            new_conv['content'][u]['keywords_3'] = kws_3
            
        new_data[list(data.keys())[c]] = new_conv
    print('Done!')
    return new_data

def keywords_stats(data):
    avg_num_kws = 0
    num_utt_without_kws = 0
    avg_num_tgram_kws = 0
    avg_num_bgram_kws = 0
    avg_num_ugram_kws = 0
    sum_num_utts = 0
    sum_num_tgram_kws = 0
    sum_num_bgram_kws = 0
    sum_num_ugram_kws = 0
    num_utts_less3_kwds = 0
    num_convs = len(list(data.keys()))
    for c in tqdm(range(num_convs)):
            conv = list(data.values())[c]['content']
            num_utts = len(conv)
            sum_num_utts+=num_utts
            for u in range(num_utts):
                    utterance = conv[u]['message']
                    #print(conv[u])
                    kws = conv[u]['keywords_3']
                    #print(kws)
                    if len(kws) ==0:
                            num_utt_without_kws+=1
                            print(utterance)
                    if len(kws) < 3 and len(kws)>0:
                            num_utts_less3_kwds+=1
                            #print(utterance)
                    tgram_kws = [kw for kw in kws if len(kw.split())==3]
                    bgram_kws = [kw for kw in kws if len(kw.split())==2]
                    ugram_kws =[kw for kw in kws if len(kw.split())==1]
                    sum_num_tgram_kws+=len(tgram_kws)
                    sum_num_bgram_kws+=len(bgram_kws)
                    sum_num_ugram_kws+=len(ugram_kws)
                    avg_num_kws+=len(kws)
    print('total number of utterances {} '.format(sum_num_utts))
    print('total number of kws {}'.format(avg_num_kws))
    print('avg number of kws per utterance {}'.format(avg_num_kws/sum_num_utts))
    print('number of utterances without keywords {}'.format(num_utt_without_kws))
    print('avg number of trigram kws per utterance {}'.format(sum_num_tgram_kws/sum_num_utts))
    print('avg number of bigram kws per utterance {}'.format(sum_num_bgram_kws/sum_num_utts))
    print('avg number of unigram kws per utterance {} {}'.format(sum_num_ugram_kws, sum_num_ugram_kws/sum_num_utts))
    print('avg number of utterance per conversation {}'.format(sum_num_utts/num_convs))
    print('number of utterances with at least one keyword and less than three keywords {}'.format(num_utts_less3_kwds))

def entity_topic_label(conv, reading,fname, entity_assign_path, data):
    if not os.path.exists('./topic_cls/heuristics/GoogleNews-vectors-negative300.magnitude'):
        print('Download word2vector Embedding')
        os.system('wget -P ./topic_cls/heuristics http://magnitude.plasticity.ai/word2vec/light/GoogleNews-vectors-negative300.magnitude')
    #print(conv, reading, entity_assign_path)
    message_en,entity2,topic_general_en,s2 = label(conv, reading,entity_assign_path)
    index = list(data.keys())
    for idx_conv, idx_label in zip(index, s2):
        assert(len(data[idx_conv]['content'])==len(idx_label))
        for i in range(len(idx_label)):
            data[idx_conv]['content'][i]['topic']=topic_general_en[idx_label[i]]
            if topic_general_en[idx_label[i]]==['General']:
                data[idx_conv]['content'][i]['entity_reading_set']=''
            else:
                data[idx_conv]['content'][i]['entity_reading_set']=entity2[idx_label[i]]
    #with open(os.path.join('alexa-prize-topical-chat-dataset',fname+'_comp.json'), 'w') as fw:
    #        json.dump(data, fw, sort_keys=False, ensure_ascii=False, indent=5)
    os.system('cd topic_cls/heuristics/;python flatten_data.py %s;python label_with_rules.py %s;python label_with_keywords.py %s;cd ../..'%(fname,fname,fname))
    label1 = csv.reader(open('./topic_cls/heuristics/data/labels_from_rules_'+fname+'.tsv','rt'),delimiter='\t')
    all_labels = {(row[0],row[1]):[row[2],json.loads(row[3]), None] for row in label1}
    reader = csv.reader(open('./topic_cls/heuristics/data/flat_data_'+fname+'.tsv', 'rt'), delimiter='\t')
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
            if data[conv_id]['content'][i]['keywords_2']==[] and data[conv_id]['content'][i]['topic']!=['General']:
                data[conv_id]['content'][i]['topic'] = ['General']
                data[conv_id]['content'][i]["entity_reading_set"] = ''
                try:
                    all_labels[(conv_id,str(conv[conv_id][i]))][-1]=['9']
                except:
                    print(conv_id,conv[conv_id][i])
                    pdb.set_trace()
    for conv_id in data:
        for i in [0,1,-2,-1]:
            if data[conv_id]['content'][i]['topic']==['General']:
                all_labels[(conv_id,str(conv[conv_id][i]))][-1]=['9']
    general_labels = all_labels
    lookup = { '1':'Fashion','2':'Politics','3':'Books','4':'Sports','5':'General Entertainment','6':'Music','7':'Science & Technology','8':'Movie' }
    keyword_labels = csv.reader(open('./topic_cls/heuristics/data/labels_from_keywords_'+fname+'.tsv', 'rt'), delimiter='\t')
    rule_labels = csv.reader(open('./topic_cls/heuristics/data/labels_from_rules_'+fname+'.tsv', 'rt'), delimiter='\t')
    header = next(rule_labels)

    for general_row, keyword_row, rule_row in zip(general_labels, keyword_labels, rule_labels):
        conv_id, idx = rule_row[0], rule_row[1]
        if data[conv_id]['content'][int(idx)]['topic'] == ['General'] :
            pass
        elif rule_row[2] == 'rule_1':
            topics = json.loads(rule_row[3])
            data[conv_id]['content'][int(idx)]['topic'] = [lookup[k] for k in topics]
        else:
            if keyword_row[4] in rule_row[3]:
                data[conv_id]['content'][int(idx)]['topic'] = [lookup[keyword_row[4]]]
            else:
                topics = json.loads(rule_row[3])
                try:
                    data[conv_id]['content'][int(idx)]['topic'] = [lookup[k] for k in topics[0]]
                except:
                    print(topics)
    return data




if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Parameters for data preprocessing')
    parser.add_argument('--device', type=str, default="cuda",
                    help='type of device to run models (cpu/cuda)')

    parser.add_argument('--data_dir', type=str, default="alexa-prize-topical-chat-dataset/conversations",
                    help='the directory including the data')

    parser.add_argument('--fname', type=str, default="test_freq",
                    help='the file name to be preprocessed')

    parser.add_argument('--mode', type=str, default="extract",
                    help='mode can be extract or stats for extraction or just getting statistics.')
    parser.add_argument('--reading', type=str, default="alexa-prize-topical-chat-dataset/reading_sets/post-build",help='the directory including the reading source data')
    parser.add_argument('--entity_assign_path', type=str, default='data_labeling/entity_topic_assign.csv',help='entity-topic mapping file')

    args = parser.parse_args()
    
    device = torch.device(args.device)


    with open(os.path.join(args.data_dir,args.fname+'.json'), 'r') as fr:
            data = json.load(fr)

    if args.mode =="extract":
        
        data = extract_keywords(data)
        data = extract_entities(data, device)
        conv = os.path.join(args.data_dir,args.fname+'.json')
        reading = os.path.join(args.reading,args.fname+'.json')
        data = entity_topic_label(conv, reading, args.fname, args.entity_assign_path, data)
        with open(os.path.join(args.data_dir,args.fname+'_comp.json'), 'w') as fw:
            json.dump(data, fw, sort_keys=False, ensure_ascii=False, indent=5)

    elif args.mode =="stats":
        print(type(data))
        entities_stats(data)
        print(type(data))
        keywords_stats(data)

