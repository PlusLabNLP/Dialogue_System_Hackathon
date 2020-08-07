from fast_bert.data_cls import BertDataBunch
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy

import torch
import sys

import logging
logging.basicConfig(level=logging.DEBUG)

DATA_PATH = sys.argv[1]
LABEL_PATH = sys.argv[2]
OUTPUT_DIR = sys.argv[3]

databunch = BertDataBunch(DATA_PATH, LABEL_PATH,
                          tokenizer='bert-base-uncased',
                          train_file='train.csv',
                          val_file='valid.csv',
                          label_file='labels.csv',
                          text_col='text',
                          label_col='label',
                          batch_size_per_gpu=64,
                          max_seq_length=50,
                          multi_gpu=False,
                          multi_label=False,
                          model_type='bert')


logger = logging.getLogger()
device_cuda = torch.device("cuda")
metrics = [{'name': 'accuracy', 'function': accuracy}]

learner = BertLearner.from_pretrained_model(
						databunch,
						pretrained_path='bert-base-uncased',
						metrics=metrics,
						device=device_cuda,
						logger=logger,
						output_dir=OUTPUT_DIR,
						finetuned_wgts_path=None,
						warmup_steps=500,
						multi_gpu=False,
						is_fp16=False,
						multi_label=False,
						logging_steps=4000)

#learner.lr_find(start_lr=1e-5,optimizer_type='lamb')

learner.fit(epochs=6,
        lr=6e-5,
        validate=True,  # Evaluate the model after each epoch
        schedule_type="warmup_cosine",
        optimizer_type="lamb")

learner.save_model()
