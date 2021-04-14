import json
from .generate import Interaction
from SETTING import parser
import torch
import time
class DialogSys(object):
    def __init__(self):
        self.parser = parser
        
        torch.manual_seed(parser['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(parser['seed'])
        #temporally comment
        self.interaction = Interaction(self.parser)

        #-----------------------------------
        
    def regenerate_keywords(self, data):
        print('regenerate keywords')
        print(data)
        histories = data['history']
        #user_utt = data['new_message'][0]
        user_utt = data['history'][-1]
        if len(histories)==1:
            prev1 = ''
            prev2 = ''
            prev_utts = '[CLS] '+prev1+' [SEP] '+prev2+' [SEP] '+user_utt+' [SEP]'
        elif len(histories)==2:
            prev1 = ''
            prev2 = histories[-2]
            prev_utts = '[CLS] '+prev1+' [SEP] '+prev2+' [SEP] '+user_utt+'[SEP]'
        else:
            prev1 = histories[-3]
            prev2 = histories[-2]
            prev_utts = '[CLS] '+prev1+' [SEP] '+prev2+' [SEP] '+user_utt+' [SEP]'
        #temporally comment
        self.interaction.temperature = float(data['temperature'])
        self.interaction.top_k = int(data['top-k'])
        self.interaction.top_p = float(data['top-p'])
        #user_utt_topic = self.interaction.get_topic(prev_utts)
        user_utt_topic = self.interaction.get_topic(prev_utts)
        user_utt_entities = self.interaction.get_entities(user_utt)
        res_utt_keywords = self.interaction.get_response_keywords(user_utt, user_utt_topic, user_utt_entities,True)
        res_utt_keywords = list(filter(lambda x: x!='', res_utt_keywords))
        #sys_utt1 = self.interaction.get_response(user_utt, user_utt_topic, res_utt_keywords)
        keywords = ' # '.join(res_utt_keywords)
        return {'responses': [''],'keywords':keywords}


        
    def decode(self, data):
        
        #histories = data['message_list']+[data['new_message'][0]] if len(data['new_message']) > 1 else data['message_list']
        histories = data['history']
        #user_utt = data['new_message'][0]
        if len(histories)==0:
            print('histories has 0 length')
            histories = ['']
        user_utt = histories[-1]#data['history'][-1]
        if len(histories)==1:
            prev1 = ''
            prev2 = ''
            prev_utts = '[CLS] '+prev1+' [SEP] '+prev2+' [SEP] '+user_utt+' [SEP]'
        elif len(histories)==2:
            prev1 = ''
            prev2 = histories[-2]
            prev_utts = '[CLS] '+prev1+' [SEP] '+prev2+' [SEP] '+user_utt+'[SEP]'
        else:
            prev1 = histories[-3]
            prev2 = histories[-2]
            prev_utts = '[CLS] '+prev1+' [SEP] '+prev2+' [SEP] '+user_utt+' [SEP]'
        #temporally comment
        self.interaction.temperature = float(data['temperature'])
        self.interaction.top_k = int(data['top-k'])
        self.interaction.top_p = float(data['top-p'])
        #user_utt_topic = self.interaction.get_topic(prev_utts)
        user_utt_topic = self.interaction.get_topic(prev_utts)
        user_utt_entities = self.interaction.get_entities(user_utt)
        res_utt_keywords = self.interaction.get_response_keywords(user_utt, user_utt_topic, user_utt_entities)
        print('before filtering',res_utt_keywords)
        #res_utt_keywords = list(filter(lambda x: x!='', res_utt_keywords))
        #print('after filtering', res_utt_keywords)
        keywords = ' # '.join(res_utt_keywords)
        print('convline', keywords)
        sys_utt = self.interaction.get_response(user_utt, user_utt_topic, res_utt_keywords)
        print('sys_utt:', sys_utt)
        res_utt_keywords = list(filter(lambda x: x!='', res_utt_keywords))
        #sys_utt1 = self.interaction.get_response(user_utt, user_utt_topic, res_utt_keywords)
        keywords = ' # '.join(res_utt_keywords)
        print('after filtering', res_utt_keywords)
        #print('after filtering sys_utt1:', sys_utt1)
        x = {'responses':[sys_utt], 'keywords': keywords}
        #--------------------------------
        
        '''
        sys_utt = 'nice thing!'
        keywords = 'apple # pear # peach'
        print('message_list', data['history'])
        print('new_message', sys_utt)
        x = {'responses':[sys_utt], 'keywords': keywords}
        time.sleep(3)
        '''
        return x

    def customize_decode(self, data):
        histories = data['history'][:-1]
        if len(histories)==0:
            user_utt=''
        else:
            user_utt = histories[-1]
        if len(histories)==0:
            prev1 = ''
            prev2 = ''
            prev_utts = '[CLS] '+prev1+' [SEP] '+prev2+' [SEP] '+user_utt+' [SEP]'
        elif len(histories)==1:
            prev1 = ''
            prev2 = histories[-1]
            prev_utts = '[CLS] '+prev1+' [SEP] '+prev2+' [SEP] '+user_utt+' [SEP]'
        else:
            prev1 = histories[-2]
            prev2 = histories[-1]
            prev_utts = '[CLS] '+prev1+' [SEP] '+prev2+' [SEP] '+user_utt+' [SEP]'
        
        self.interaction.temperature = float(data['temperature'])
        self.interaction.top_k = int(data['top-k'])
        self.interaction.top_p = float(data['top-p'])
        #user_utt_topic = self.interaction.get_topic(prev_utts)
        user_utt_topic = self.interaction.get_topic(user_utt)
        user_utt_entities = self.interaction.get_entities(user_utt)
        res_utt_keywords = data['keywords']
        print('regenerate before append:',res_utt_keywords)
        res_utt_keywords.append('')
        print('regenerate after append:',res_utt_keywords)
        sys_utt = self.interaction.get_response(user_utt, user_utt_topic, res_utt_keywords)
        print('sys_utt:', sys_utt)
        res_utt_keywords = list(filter(lambda x: x!='', res_utt_keywords))
        #sys_utt1 = self.interaction.get_response(user_utt, user_utt_topic, res_utt_keywords)
        #print('sys_utt1:', sys_utt1)
        print('regenrate filtering', res_utt_keywords)
        res_utt_keywords = " # ".join(res_utt_keywords)
        x = {'responses':[sys_utt], 'keywords': res_utt_keywords}
        '''
        print(prev_utts)
        
        sys_utt = 'nice thing!'
        keywords = 'apple # pear # peach'
        print(keywords)
        print('message_list', data['history'])
        print('new_message', sys_utt)
        
        time.sleep(3)
        '''
        return x

    def baseline(self, data):
        print('baseline generate')
        #tokenizer = self.baseline_tokenizer
        #model = self.baseline_model
        #print(data)
        histories = data['history']
        if len(histories)>0:
            user_utt = data['history'][-1]
            utterance = self.interaction.baseline_decode(user_utt)
        else:
            utterance = self.interaction.baseline_decode('BEGIN')
        return {'responses': [utterance], 'keywords' : ''}
