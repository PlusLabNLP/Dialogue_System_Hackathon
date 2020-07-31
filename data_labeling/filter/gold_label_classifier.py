#from rake_nltk import Rake
import pandas as pd
import ast
import numpy as np 
#from sklearn.svm import SVC
from pymagnitude import *
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import copy
import pandas as pd
import argparse
def main(dataset, keywords, dst1, dst2, embedding_path):
    df = pd.read_csv(dataset,index_col=0)
    sentence = df['sentence'].values.tolist()
    entity = df['entity'].values.tolist()
    topic1 = df['topic'].values.tolist()
    #label = df["entity: accetable: 1 / nonacceptable: 0"].values.tolist()
    #r =Rake()
    '''
    for i in range(len(entity)):
        entity[i] = ast.literal_eval(entity[i])


    for i in range(len(entity)):
        entity[i] = entity[i][:-1]
    '''
    for i in range(len(topic1)):
        topic1[i] = ast.literal_eval(topic1[i])
    for i in range(len(entity)):
        try:
            tmp = copy.copy(entity[i])
            entity[i] = ast.literal_eval(entity[i])
        except:
            print(tmp)
    #clf = SVC(C=0.9, kernel = 'rbf',gamma='scale',random_state=42)
    #vectors = Magnitude("/nas/home/zixiliu/topic_ext/word2vec/GoogleNews-vectors-negative300.magnitude")
    vectors = Magnitude(embedding_path)

    topic = {1:'Fashion',2:'Politics', 3:'Books', 4:'Sports',5:'General Entertainment',6:'Music',7:'Science & Technology', 8:'Movie'}
    assignment = pd.read_csv('entity_topic_assign.csv')
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

    pred_entity = []
    pred_topic = []
    with open(keywords,'r') as f: 
        for idx,i in tqdm(enumerate(f)): 
            dist = 0
            tmp = i.strip() 
            if tmp=='': 
                tmp = sentence[idx] 
                tmp = word_tokenize(tmp) 
            else: 
                tmp = tmp.split(',') 
                tmp.remove('') 
                tmp = word_tokenize(tmp[0]) 
            tmp = vectors.query(tmp)
            match_ent = ''
            for ent in en_label:
                dist_tmp = np.linalg.norm(np.matmul(tmp,en_label[ent].T))
                if dist_tmp>dist:
                    dist = dist_tmp
                    match_ent = ent
            #print(match_ent)
            #print(entity[idx])
            pred_entity.append(match_ent)
            pred_topic.append(entity_assign[match_ent])
            #print(dist)

    #print(topic1)
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
        #if pred_entity[i]!=entity[i][0] and label[i]==0:
        #    num1+=1
        #if pred_topic[i]!=topic1[i][0] and label[i]==0:
        #    num2+=1
    print(num1)
    print(num2)
    print(num1/len(sentence))
    print(num2/len(sentence))
    df = pd.DataFrame(data1).to_csv(dst1)
    df = pd.DataFrame(data2).to_csv(dst2)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("keywords")
    parser.add_argument("dst1")
    parser.add_argument("dst2")
    args = parser.parse_args()
    main(args.dataset, args.keywords, args.dst1, args.dst2)