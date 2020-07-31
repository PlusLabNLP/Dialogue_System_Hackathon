from rake_nltk import Rake
import pandas as pd
import ast
import argparse
#import numpy as np 
#from sklearn.svm import SVC
#from pymagnitude import *
def main(src, dst):
    df = pd.read_csv(src,index_col=0)
    sentence = df['sentence'].values.tolist()
    #topic = df['entity'].values.tolist()
    #topic = df['topic_with_general'].values.tolist()
    #label = df["entity: accetable: 1 / nonacceptable: 0"].values.tolist()
    r =Rake()
    #for i in range(len(entity)):
    #    entity[i] = ast.literal_eval(entity[i])

    #for i in range(len(topic)):
    #    topic[i] = ast.literal_eval(topic[i])
    #for i in range(len(entity)):
    #    entity[i] = entity[i][:-1]
    #clf = SVC(C=0.9, kernel = 'rbf',gamma='scale',random_state=42)
    #vectors = Magnitude("/nas/home/zixiliu/topic_ext/word2vec/GoogleNews-vectors-negative300.magnitude")
    #dist = []
    '''
    Generate Rake keywords and save in keywords.txt
    '''
    #r.extract_keywords_from_sentences(sentence)

    with open(dst,'w') as f:
        for i in sentence:
            r.extract_keywords_from_text(i)
            keywords = r.get_ranked_phrases()
            for j in keywords:
                f.write(j+',')
            f.write('\n')

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("dst")
    args = parser.parse_args()
    main(args.src, args.dst)