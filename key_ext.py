import json
import yake
from tqdm import tqdm

fname='train'


n_gram =3
kw_extractor = yake.KeywordExtractor(n=n_gram)
special_inputs =[':)', '.']

def sortFunc(e):
	return len(e.split())

#configuration1:extract trigram, bigram and unigram without overlapping
def get_kwd_1(keywords):
	outputs = []
	keys = [keyword for keyword, score in keywords]
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
	keys = [keyword for keyword, score in keywords]
	return keys[:3]


#configuration3: get the unigram keywords only
def get_kwd_3(keywords):
	outputs = []
	keys = [keyword for keyword, score in keywords]
	for key in keys:
		if len(key.split()) == 1:
			outputs.append(key)

	return outputs


def extract_kws(data):
	num_convs = len(list(data.keys()))
	print('Extracting keywords for all utterances of {} conversation'.format(num_convs))
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
	with open('alexa-prize-topical-chat-dataset/conversations/{}_ext.json'.format(fname), 'w') as fw:
		json.dump(new_data, fw, indent=5)


def get_stats(convs):
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


with open('alexa-prize-topical-chat-dataset/conversations/{}.json'.format(fname), 'r') as fr:
	data = json.load(fr)

extract_kws(data)
#get_stats(data)

