from fast_bert.prediction import BertClassificationPredictor
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import sys
import csv
from os import path
from tqdm import tqdm
import pdb
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('-o','--output_dir',type=str,help='inference output dir',default='./inference')
parser.add_argument('-m','--model_dir',type=str,help="the directory of 'model_out' folder",default='./')
parser.add_argument('-l','--label_dir',type=str,help='the directory of label.csv file', default='./')
parser.add_argument('-i','--input_name',type=str,help='choose from valid, test',default='test')
parser.add_argument('-b','--batch_size',type=int,help='inference batch size',default=128)
parser.add_argument('--show_confusion_matrix',type=bool,help='whether to show confusion matrix', default=True)
parser.add_argument('--show_class_accuracy',type=bool,help='whether to show class accuracy',default=True)
args = parser.parse_args()
MODEL_DIR = args.model_dir#sys.argv[1]

MODEL_PATH = path.join(MODEL_DIR, 'model_out')

predictor = BertClassificationPredictor(
        model_path=MODEL_PATH,
        label_path=args.label_dir,  #sys.argv[2], # directory for labels.csv file
        multi_label=False,
        model_type='bert',
        do_lower_case=True)
INPUT = os.path.join('data',args.input_name+'.csv')
texts = list(csv.reader(open(INPUT, 'rt'))) # sys.argv[3]
batchsize = args.batch_size
multiple_predictions = []
for i in tqdm(range(1,len(texts),batchsize)):
    batch_texts = []
    if i+batchsize>len(texts):
        for j in range(i,len(texts)):
            batch_texts.append(texts[j][0])
        tmp_pred = predictor.predict_batch(batch_texts)
        multiple_predictions.extend(tmp_pred)
    else:
        for j in range(i,i+batchsize):
            batch_texts.append(texts[j][0])
        tmp_pred = predictor.predict_batch(batch_texts)
        multiple_predictions.extend(tmp_pred)
#multiple_predictions = predictor.predict_batch(i[0] for i in texts[1:])
if not os.path.exists(args.output_dir):
    os.system('mkdir '+args.output_dir)
with open(os.path.join(args.output_dir,args.input_name+'-inference.txt'), 'wt') as fh:#sys.argv[4]
    arg_maxes = [ i[0][0]+'\n' for i in multiple_predictions ]
    fh.writelines(arg_maxes)

# report accuracy
#print(texts[0])
gold = [ i[1] for i in texts[1:] ]
accuracy = sum([ i.strip() == j.strip() for i, j in zip(gold, arg_maxes) ]) / len(gold)
print('Accuracy: %f' % accuracy)
for i in range(len(gold)):
    gold[i] = int(gold[i].strip())

for i in range(len(arg_maxes)):
    arg_maxes[i] = int(arg_maxes[i].strip())

classes = ['1','2','3','4','5','6','7','8','9']
if args.show_confusion_matrix:
    confusionMatrix = confusion_matrix(gold, arg_maxes)
    confusionMatrix1 = confusion_matrix(gold,arg_maxes,normalize='true')
    print('*****confusion matrix*****')
    print(confusionMatrix)
    print('*****confusion matrix (Accuracy)*****')
    print(confusionMatrix1)
    group_counts = ["{0:0.0f}".format(value) for value in confusionMatrix.flatten()]
    group_persentages = ["{0:.2%}".format(value) for value in confusionMatrix1.flatten()]
    labels = [f"{v1}\n{v2}" for v1,v2 in zip(group_counts,group_persentages)]
    labels = np.asarray(labels).reshape(9,9)
    hm = sns.heatmap(confusionMatrix,annot=labels,fmt='',cmap='Blues',xticklabels=classes,yticklabels=classes)
    figure = hm.get_figure()
    figure.savefig('./inference/confusion_matrix.png')
    
class_acc = []
for i in range(9):
    class_acc.append(confusionMatrix1[i,i])
if args.show_class_accuracy:
    print('*****Class Accuracy*****')
    print(class_acc)
