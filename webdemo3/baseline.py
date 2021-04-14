from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
tokenizer = AutoTokenizer.from_pretrained("/nas/home/zixiliu/dialogsystem_model/baseline")
model = AutoModelForCausalLM.from_pretrained("/nas/home/zixiliu/dialogsystem_model/baseline")
model = model.cuda()
#f = open('test_response1.txt','w')
a = []
prev = 'Hi! Do you work out?'
new_user_input_ids = tokenizer.encode(prev + tokenizer.eos_token, return_tensors='pt',max_length=128).cuda()
chat_history_ids = model.generate(new_user_input_ids, top_k=10, top_p=0.70, max_length=50, pad_token_id=tokenizer.eos_token_id)
utterance = tokenizer.decode(chat_history_ids[0], skip_special_tokens=True)
print(utterance)
'''
for line in open('./examples/dialog/test.txt'):
        if 'BEGIN' not in line:
                try:
                        prev , nex = line.strip().split(' <EOT> ')[0], line.strip().split(' <EOT> ')[1]
                        new_user_input_ids = tokenizer.encode(prev + tokenizer.eos_token, return_tensors='pt',max_length=128).cuda()
                        chat_history_ids = model.generate(new_user_input_ids, max_length=128, pad_token_id=tokenizer.eos_token_id)
                        utterance = tokenizer.decode(chat_history_ids[0], skip_special_tokens=True)
                        if ' <EOT> '  in utterance:
                                f.write(prev+ ' <EOT> '+utterance.split(' <EOT> ')[1]+'\n')
                        else:
                                f.write(prev+ ' <EOT> '+utterance+'\n')
                except:
                        f.write(prev+ ' <EOT> '+"Failed to generate"+'\n')
        else:
                step = 0
                prev = 'BEGIN'
                f.write(line)
'''

