from fast_bert.prediction import BertClassificationPredictor

import sys
import csv
from os import path

OUTPUT_DIR = sys.argv[1]
MODEL_PATH = path.join(OUTPUT_DIR, 'model_out')

predictor = BertClassificationPredictor(
        model_path=MODEL_PATH,
        label_path=sys.argv[2], # location for labels.csv file
        multi_label=False,
        model_type='bert',
        do_lower_case=True)

texts = list(csv.reader(open(sys.argv[3], 'rt')))
multiple_predictions = predictor.predict_batch(i[1] for i in texts)

with open(sys.argv[4], 'wt') as fh:
    arg_maxes = [ i[0][0]+'\n' for i in multiple_predictions ]
    fh.writelines(arg_maxes)

# report accuracy
print(texts[0])
gold = [ i[2] for i in texts ]
accuracy = sum([ i.strip() == j.strip() for i, j in zip(gold, arg_maxes) ]) / len(gold)
print('Accuracy: %f' % accuracy)
