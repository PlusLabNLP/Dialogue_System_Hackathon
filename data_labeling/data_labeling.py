import json
import copy
import pandas as pd
import re
#from rake_nltk import Rake
import numpy as np 
#from pymagnitude import *
from nltk.tokenize import word_tokenize
from tqdm import tqdm


def keywords_ext(sentence):
    #df = pd.read_csv(src,index_col=0)
    #sentence = df['sentence'].values.tolist()
    r = Rake()
    '''
    Generate Rake keywords and save in keywords.txt
    '''
    keyword_all = []
    for i in sentence:
        r.extract_keywords_from_text(i)
        keywords = r.get_ranked_phrases()
        keyword_all.append(keywords)
    return keyword_all

def gold_label_classifier(sentence, entity, topic1, keywords, embedding_path, entity_assign_path):
    #df = pd.read_csv(dataset,index_col=0)
    #sentence = df['sentence'].values.tolist()
    #entity = df['entity'].values.tolist()
    #topic1 = df['topic'].values.tolist()
    '''
    for i in range(len(entity)):
        entity[i] = ast.literal_eval(entity[i])
    for i in range(len(entity)):
        entity[i] = entity[i][:-1]
    '''
    #for i in range(len(topic1)):
    #    topic1[i] = ast.literal_eval(topic1[i])
    #for i in range(len(entity)):
    #    try:
    #        tmp = copy.copy(entity[i])
    #        entity[i] = ast.literal_eval(entity[i])
    #    except:
    #        print(tmp)
    vectors = Magnitude(embedding_path)

    topic = {1:'Fashion',2:'Politics', 3:'Books', 4:'Sports',5:'General Entertainment',6:'Music',7:'Science & Technology', 8:'Movie'}
    assignment = pd.read_csv(entity_assign_path)
    entity_assign = dict()
    for index, row in assignment.iterrows():
        tmp = row['topic']
        tmp = topic[int(tmp)]
        entity_assign[row['entity']] = tmp
        
    en_label = {}
    for i in entity_assign:
        tmp = i.replace('(','')
        tmp = tmp.replace(')','')
        tmp = word_tokenize(tmp)
        en_label[i] = vectors.query(tmp)[-1]

    pred_entity = [-1 for i in range(len(sentence))]
    pred_topic = [-1 for i in range(len(sentence))]
    for idx,i in tqdm(enumerate(keywords)): 
        dist = 0
        tmp = i
        if tmp==[''] or len(tmp)==0: 
            tmp = sentence[idx]
            tmp = word_tokenize(tmp)
        else: 
            #tmp = tmp.split(',') 
            #tmp.remove('') 
            tmp = word_tokenize(tmp[0])
        tmp = vectors.query(tmp)
        match_ent = ''
        for ent in en_label:
            dist_tmp = np.linalg.norm(np.matmul(tmp,en_label[ent].T))
            if dist_tmp>dist:
                dist = dist_tmp
                match_ent = ent
        if match_ent in entity[idx]:
            pred_entity[idx]=1
        else:
            pred_entity[idx]=0
        if entity_assign[match_ent] in topic1[idx]:
            pred_topic[idx]=1
        else:
            pred_topic[idx]=0
        if topic1[idx][0]=='General':
            pred_topic[idx]=1
        #pred_entity.append(match_ent)
        #pred_topic.append(entity_assign[match_ent])
    return pred_topic, pred_entity
    '''
    num1 = 0
    num2 = 0
    data1 = {'sentence':[],'entity':[],'topic':[]}
    data2 = {'sentence':[],'entity':[],'topic':[]}
    for i in range(len(sentence)):
        if pred_entity[i] in entity[i]:
            data1['sentence'].append(sentence[i])
            data1['entity'].append(entity[i])
            data1['topic'].append(topic1[i])
            num1+=1
        if pred_topic[i] in topic1[i]:
            data2['sentence'].append(sentence[i])
            data2['entity'].append(entity[i])
            data2['topic'].append(topic1[i])
            num2+=1

    print(num1)
    print(num2)
    print(num1/len(sentence))
    print(num2/len(sentence))
    df = pd.DataFrame(data1).to_csv(dst1)
    df = pd.DataFrame(data2).to_csv(dst2)
    '''

def label(conv, reading,entity_assign_path):#, embedding_path):
    topic = {1:'Fashion',2:'Politics', 3:'Books', 4:'Sports',5:'General Entertainment',6:'Music',7:'Science & Technology', 8:'Movie'}
    assignment = pd.read_csv(entity_assign_path)
    entity_assign = dict()
    for index, row in assignment.iterrows():
        tmp = row['topic']
        tmp = topic[int(tmp)]
        entity_assign[row['entity']] = tmp


    #src1 = 'conversations/train.json'
    #src2 = 'reading_sets/post-build/train.json'
    src1 = conv
    src2 = reading
    with open(src1,'r') as f, open(src2,'r') as g:
        t1 = json.load(f)
        t2 = json.load(g)
    dataset = []
    #print('Total number of dialog: ',len(t1.keys()))
    num1=0
    for i in t1.keys():
        num1 = num1+1
        t1[i].pop("article_url")
        t1[i].pop('config')
        t1[i].pop('conversation_rating')
        for j in range(len(t1[i]['content'])):
            agent = t1[i]['content'][j]['agent']
            source = t1[i]['content'][j]['knowledge_source']
            t1[i]['content'][j]['entity'] = []
            t1[i]['content'][j]['topic'] = []
            source = set(source)
            set_fs = set(['FS1','FS2','FS3'])
            set_as = set(['AS1','AS2','AS3'])
            if set_fs&source!=set():
                for k in list(set_fs&source):
                    t1[i]['content'][j]['entity'].append(t2[i][agent][k]['entity'])
            elif set_fs&source==set() and set_as&source!=set():
                for k in list(set_as&source):
                    tmp = t2[i]['article'][k]
                    t1[i]['content'][j]['entity'].append("No entity: id: "+i+'; '+k+': '+tmp)
        for v in range(len(t1[i]['content'])):
            num = v
            if t1[i]['content'][v]['entity']==[]:
                set1 = set(t1[i]['content'][num-1]['knowledge_source'])&set(['FS1','FS2','FS3'])
                if num<len(t1[i]['content'])-1:
                    set2 = set(t1[i]['content'][num+1]['knowledge_source'])&set(['FS1','FS2','FS3'])
                else:
                    set2 = set(t1[i]['content'][num-1]['knowledge_source'])&set(['FS1','FS2','FS3'])
                if len(set1&set2)!=0:
                    source = [x for x in set1&set2]
                    for u in range(len(source)):
                        t1[i]['content'][v]['entity'].append(t2[i][t1[i]['content'][v]["agent"]][source[u]]['entity'])
                    t1[i]['content'][v]['entity'].append('INFER')
                else:
                    tmp = []
                    source = [x for x in set1|set2]
                    if len(source)>0:
                        for u in range(len(source)):
                            tmp.append(t2[i][t1[i]['content'][v]["agent"]][source[u]]['entity'])
                        t1[i]['content'][v]['entity'] = ["No entity: topic change",tmp]
                    else:
                        t1[i]['content'][v]['entity'] = ["No entity: cannot infer from former and latter"]
        dataset.append([])
        dataset[-1] = copy.deepcopy(t1[i])


    # for utterance without topics, the assign the name entity to that appear the most. If a dialog don't have topic, assign all topics of entities
    for i in range(len(dataset)):
        tmp = {}
        for k in range(len(dataset[i]['content'])):
            if 'No entity' in dataset[i]['content'][k]['entity'][0]:
                continue
            for j in range(len(dataset[i]['content'][k]['entity'])):
                if dataset[i]['content'][k]['entity'][j]=='INFER':
                    pass
                else:
                    #print(dataset[i]['content'][k]['entity'])
                    try:
                        dataset[i]['content'][k]['topic'].append(entity_assign[dataset[i]['content'][k]['entity'][j]])
                    except:
                        print(dataset[i]['content'][k]['entity'])
                    if entity_assign[dataset[i]['content'][k]['entity'][j]] not in tmp:
                        tmp[entity_assign[dataset[i]['content'][k]['entity'][j]]] = 1
                    else:
                        tmp[entity_assign[dataset[i]['content'][k]['entity'][j]]] = tmp[entity_assign[dataset[i]['content'][k]['entity'][j]]]+1
        if tmp!={}:
            max_key = max(tmp, key=tmp.get)
            keys = []
            for v in tmp:
                if tmp[v]==tmp[max_key]:
                    keys.append(v)
        else:
            keys = []
            for k in range(len(dataset[i]['content'])):
                if 'No entity: id: ' in dataset[i]['content'][k]['entity'][0]:
                    idx = re.findall(r'No entity: id: (.+); AS',dataset[i]['content'][k]['entity'][0])
            for tt in ['FS1','FS2','FS3']:
                keys.append(entity_assign[t2[idx[0]]['agent_1'][tt]['entity']])
            '''
            print(i)
            print(json.dumps(dataset[i]['content'],indent=4))
            max_key = max(tmp, key=tmp.get)
            keys = []
            for v in tmp:
                if v==max_key:
                    keys.append(v)
            '''
        for k in range(len(dataset[i]['content'])):
            if dataset[i]['content'][k]['topic']==[]:
                if dataset[i]['content'][k]['entity'][0]=='No entity: topic change': 
                    for j in dataset[i]['content'][k]['entity'][1]:
                        dataset[i]['content'][k]['topic'].append(entity_assign[j])
                else:
                    for v in keys:
                        dataset[i]['content'][k]['topic'].append(v)

    sentence = []
    NER = []
    topic = []
    for i in dataset:
        for j in i['content']:
            sentence.append(j['message'])
            NER.append(j['entity'])
            topic.append(j['topic'])
    data = {'sentence':sentence,'entity':NER,'topic':topic}
    train = pd.DataFrame(data)  
    #df.to_csv('train.csv',encoding='utf-8')
    all_topic = copy.deepcopy(topic)
    all_entity = copy.deepcopy(NER)
    message = copy.deepcopy(sentence)

    train['topic_with_general'] = None
    train['topic_as'] = None
    topic_as = []
    number3 = 0 # count the number of adjusting AS to previous appeared entity
    number4 = 0 # count the number of all AS
    for i in range(len(all_entity)):
        tmp = all_entity[i]
        t1 = True
        t2 = True
        if 'No entity: id: ' in tmp[0]:
            number4+=1
            k1 = i-1
            k2 = i+1
            if 'No entity: cannot infer from former and latter' in all_entity[k1][0] or 'No entity: id: ' in all_entity[k1][0]:
                t1 = False
            if 'No entity: cannot infer from former and latter' in all_entity[k2][0] or 'No entity: id: ' in all_entity[k2][0]:
                t2 = False
            if t1 or t2:
                number3+=1
                topic_as.append(list(set(all_topic[k1]+all_topic[k2])))
                #print(list(set(all_topic[k1]+all_topic[k2])))
                #print(tmp)
                #print(all_entity[k1])
                #print(all_entity[k2])
            else:
                topic_as.append(list(set(all_topic[i])))
                #print(list(set(all_topic[k1]+all_topic[k2])))
                #print(tmp)
                #print(all_entity[k1])
                #print(all_entity[k2])
        else:
            topic_as.append(list(set(all_topic[i])))

    src1 = conv
    src2 = reading
    with open(src1,'r') as f, open(src2,'r') as g:
        t1 = json.load(f)
        t2 = json.load(g)
    num1 = -1
    num2 = []
    for i in t1.keys():
        #num1 = num1+1
        #print(t1[i])
        t1[i].pop("article_url")
        t1[i].pop('config')
        for j in range(len(t1[i]['content'])):
            num1 = num1+1
            if j<1 or j>(len(t1[i]['content'])-3):
                num2.append(num1)
    num2 = set(num2)

    src1 = conv
    src2 = reading
    with open(src1,'r') as f, open(src2,'r') as g:
        t1 = json.load(f)
        t2 = json.load(g)
    s1 = -1
    s2 = []
    for i in t1.keys():
        #num1 = num1+1
        #print(t1[i])
        t1[i].pop("article_url")
        t1[i].pop('config')
        temp = []
        for j in range(len(t1[i]['content'])):
            s1 = s1+1
            #if j<1 or j>(len(t1[i]['content'])-3):
            temp.append(s1)
        s2.append([])
        s2[-1] = copy.copy(temp)

    topic_with_general = []
    change = []
    ext = {'Fashion':['fashion'],
        'Politics':['politics','us','united state','vote','republican','govern',"gov't",'obama','leader','senate','summit','democratic'], 
        'Books':['book','shakespeare','isaac asimov','robot','read','stan lee','libraries','library','hero','character','novel'], 
        'Sports':['sport','nba'],
        'General Entertainment':['game','entertain'],
        'Music':['rap','sing','song','music'],
        'Science & Technology':['ocean','water','earth','programming language'], 
        'Movie':['movie','imdb','superhero','actor','Oscars','film']}
    for i in range(len(all_entity)):
        if i in num2:
            if 'No entity: cannot infer from former and latter' in all_entity[i][0] or 'No entity: id: ' in all_entity[i][0]:
                topic_with_general.append(topic_as[i])
            elif 'No entity: topic change'==all_entity[i][0]:
                check = False
                tmp_sent = message[i].lower()
                # {1:'Fashion',2:'Politics', 3:'Books', 4:'Sports',5:'General Entertainment',6:'Music',7:'Science & Technology', 8:'Movie'}
                ext_temp = []
                for x in topic_as[i]:
                    ext_temp = ext_temp+ext[x]
                for j in all_entity[i][1]+ext_temp:
                    j = j.lower()
                    j=j.split('(')[0].strip()
                    temp = j.split()
                    temp = list(filter(lambda y:y!='',temp))
                    for u in temp:
                        if u in tmp_sent:
                            check = True
                if check==False:
                    topic_with_general.append(['General'])
                    change.append(i)
                else:
                    topic_with_general.append(topic_as[i])
            else:
                check = False
                tmp_sent = str(message[i]).lower()
                ext_temp = []
                for x in topic_as[i]:
                    ext_temp = ext_temp+ext[x]
                for j in all_entity[i]+ext_temp:
                    j = j.lower()
                    j=j.split('(')[0].strip()
                    temp = j.split()
                    temp = list(filter(lambda y:y!='',temp))
                    for u in temp:
                        if u in tmp_sent:
                            check = True
                if check==False:
                    topic_with_general.append(['General'])
                    change.append(i)
                else:
                    topic_with_general.append(topic_as[i])
        else:
            topic_with_general.append(topic_as[i])

    message_en=[]
    entity_en=[]
    topic_en=[]
    topic_as_en=[]
    topic_general_en=[]
    check = []
    rule2 = []
    for i in s2:
        for j in i:
            if 'No entity: topic change' in all_entity[j][0]:
                check.append(j)
                message_en.append(copy.copy(message[j]))
                entity_en.append(list(set(all_entity[j][1])))
                topic_en.append(copy.copy(all_topic[j]))
                topic_as_en.append(copy.copy(topic_as[j]))
                topic_general_en.append(copy.copy(topic_with_general[j]))
            elif 'INFER' in all_entity[j]:
                check.append(j)
                rule2.append(j)
                message_en.append(copy.copy(message[j]))
                entity_en.append(copy.copy(all_entity[j][:-1]))
                topic_en.append(copy.copy(all_topic[j]))
                topic_as_en.append(copy.copy(topic_as[j]))
                topic_general_en.append(copy.copy(topic_with_general[j]))
            else:
                check.append(j)
                message_en.append(copy.copy(message[j]))
                entity_en.append(copy.copy(all_entity[j]))
                topic_en.append(copy.copy(all_topic[j]))
                topic_as_en.append(copy.copy(topic_as[j]))
                topic_general_en.append(copy.copy(topic_with_general[j]))

    entity_freq = [[] for i in range(len(s2))]
    entity_except = []
    for i in range(len(s2)):
        tmp = {}
        for j in s2[i]:
            if 'No entity: ' not in all_entity[j][0] and 'INFER' != all_entity[j][-1]:
                for k in all_entity[j]:
                    if k not in tmp:
                        tmp[k] = 1
                    if k in tmp:
                        tmp[k]+=1
        if tmp=={}:
            entity_except.append(i)
            entity_freq[i] = []
        else:
            max_key = max(tmp, key=tmp.get)
            keys = []
            for v in tmp:
                if tmp[v]==tmp[max_key]:
                    keys.append(v)
            entity_freq[i] = keys

    check_rule4_0 = []
    check_rule4_1 = [[],[]]
    check_rule3 = []
    entity2 = copy.deepcopy(entity_en)
    number3 = 0 # count the number of adjusting AS to previous appeared entity
    number4 = 0 # count the number of all AS
    for i in range(len(entity_en)):
        tmp = all_entity[i]
        t1 = True
        t2 = True
        if 'No entity: id: ' in tmp[0]:
            number4+=1
            k1 = i-1
            k2 = i+1
            if 'No entity: cannot infer from former and latter' in all_entity[k1][0] or 'No entity: id: ' in all_entity[k1][0]:
                t1 = False
            if 'No entity: cannot infer from former and latter' in all_entity[k2][0] or 'No entity: id: ' in all_entity[k2][0]:
                t2 = False
            if t1 or t2:
                # Rule5
                number3+=1
                check_rule4_1[0].append(i)
                
                temp2 = list(set(entity2[k1]+entity2[k2]))
                temp2 = list(filter(lambda x:'No entity:'not in x,temp2))
                entity2[i] = copy.copy(temp2)
            else:
                # Rule6
                check_rule4_1[1].append(i)
                for idx, diag in enumerate(s2):
                    if i in diag:
                        entity2[i] = copy.copy(entity_freq[idx])
                        break
        elif 'No entity: cannot infer from former and latter' in tmp[0]:
            # Rule4
            check_rule4_0.append(i)
            for idx, diag in enumerate(s2):
                if i in diag:
                    entity2[i] = copy.copy(entity_freq[idx])
                    break
        elif 'No entity: topic change' in tmp[0]:
            # Rule3
            check_rule3.append(i)

    rule1 = []
    infer_set = set(rule2+check_rule3+check_rule4_0+check_rule4_1[0]+check_rule4_1[1])
    for i in range(len(all_entity)):
        if i not in infer_set:
            rule1.append(i)

    #keywords = keywords_ext(message_en)
    #pred_entity,pred_topic = gold_label_classifier(message_en,entity2,topic_general_en,keywords,embedding_path,entity_assign_path)
    return message_en,entity2,topic_general_en,s2#pred_entity,pred_topic,s2


