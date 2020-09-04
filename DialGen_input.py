import torch
import transformers
from transformers import AutoModelWithLMHead, AutoTokenizer
from tqdm import tqdm
import os
import argparse
import json


def get_data(data_dir, fname):


	with open(os.path.join(data_dir,fname+'.json'), 'r') as fr:
		data = json.load(fr)

	output =[]
	for key in tqdm(data):
		conv = data[key]["content"]
		prev_utt = 'BEGIN'
		prev_topic = 'BEGIN'
		for utt in conv:
			utt_msg = utt["message"].strip()
			if '\n' in utt_msg:
				#print(utt_msg)
				utt_msg = utt_msg.replace('\n', '')
			utt_topic = ' # '.join(utt["topic"]).strip()
			utt_keys =  ' # '.join(utt["keywords_1"]).strip()
			if utt_keys =='':
				utt_keys = 'NULL'
			output.append(prev_topic + ' <EOT> ' + prev_utt + ' <EOU> ' + utt_keys + ' <EOK> ' + utt_msg)
			prev_utt = utt_msg
			prev_topic = utt_topic
			
	return output






if __name__=="__main__":

	parser = argparse.ArgumentParser(description='Parameters for dialogue generation module')
	parser.add_argument('--device', type=str, default="cuda",
                    help='type of device to run models (cpu/cuda)')
	parser.add_argument('--data_dir', type=str, default="alexa-prize-topical-chat-dataset/conversations",
                    help='the directory including the data')
	parser.add_argument('--fn', type=str, default="train",
                    help='the name of file')
	parser.add_argument('--mode', type=str, default="preprocess",
                    help='preprocess/train/generation')
	args = parser.parse_args()
    
	if args.mode == 'preprocess':
		if args.fn =='test' or args.fn =='valid':
			fname=args.fn+'_freq_comp'
			output1_lines = get_data(args.data_dir, fname)
			fname=args.fn +'_rare_comp'
			output2_lines = get_data(args.data_dir, fname)
			output_lines = output1_lines + output2_lines

		else:
			fname=args.fn+'_comp'
			output_lines = get_data(args.data_dir, fname)

		fw = open(os.path.join(args.data_dir, 'DialGen_{}_case2_kwcase1.txt'.format(args.fn)), 'w')
		fw.writelines("%s\n" % line for line in output_lines)

	




    
