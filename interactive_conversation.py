import torch
from transformers import AutoModelWithLMHead, AutoTokenizer,AutoModelForTokenClassification
from fast_bert.prediction import BertClassificationPredictor
import argparse 
import numpy as np
from fairseq.models.bart import BARTModel
from tqdm import trange
import torch.nn.functional as F

class Interaction():

    def __init__(self, args):
        self.gen_model_type = args.gen_model_type
        self.gen_model_path = args.gen_model_path
        self.conv_line_path = args.conv_line_path
        self.gen_length = args.length
        self.temperature = args.temperature
        self.top_k = args.top_k
        self.top_p = args.top_p
        self.stop_token= args.stop_token
        self.repetition_penalty= args.repetition_penalty
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.gen_model_type = self.gen_model_type.lower()
        self.lookup = { '1':'Fashion','2':'Politics','3':'Books','4':'Sports','5':'General Entertainment','6':'Music','7':'Science & Technology','8':'Movie','9':'General' }
        self.topic_cls = BertClassificationPredictor(
            model_path=args.topic_cls_path,
            label_path=args.label_dir,  #sys.argv[2], # directory for labels.csv file
            multi_label=False,
            model_type='bert',
            do_lower_case=True)

        self.entity_ext_model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
        self.entity_ext_model.to(self.device)
        self.entity_ext_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


        if self.gen_model_type =='dialogpt':
            self.gen_tokenizer = AutoTokenizer.from_pretrained(self.gen_model_path)
            self.gen_model = AutoModelWithLMHead.from_pretrained(self.gen_model_path)
            self.gen_model.cuda()
            self.gen_model.eval()
        
        self.conv_line =  BARTModel.from_pretrained(self.conv_line_path,checkpoint_file='checkpoint_best.pt',data_name_or_path=self.conv_line_path)
        self.conv_line.cuda()
        self.conv_line.eval()
        
 
    def get_topic(self, utterance):
        '''
        this method calls the topic cls and returns utterace's topic
        '''
        topic = self.lookup[self.topic_cls.predict(utterance)[0][0]]
        return topic

    def get_entities(self, utterance):
        '''
        this method calls the entity extractor model and returns utterace's entities
        '''
        entities =''

        label_list = [
        "O",       # Outside of a named entity
        "B-MISC",  # Beginning of a miscellaneous entity right after another miscellaneous entity
        "I-MISC",  # Miscellaneous entity
        "B-PER",   # Beginning of a person's name right after another person's name
        "I-PER",   # Person's name
        "B-ORG",   # Beginning of an organisation right after another organisation
        "I-ORG",   # Organisation
        "B-LOC",   # Beginning of a location right after another location
        "I-LOC"    # Location
        ]

        # Bit of a hack to get the tokens with the special tokens
        tokens = self.entity_ext_tokenizer.tokenize(self.entity_ext_tokenizer.decode(self.entity_ext_tokenizer.encode(utterance)))
        inputs = self.entity_ext_tokenizer.encode(utterance, return_tensors="pt").to(self.device)
        
        outputs = self.entity_ext_model(inputs)[0]
        predictions = torch.argmax(outputs, dim=2)

        entity = [(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].tolist())]

        # delete '##' before tokens
        r = []
        r_tags = []
        for i, tpl in enumerate(entity):
            if tpl[0].startswith("##"):
                if r:
                    r[-1] += tpl[0][2:]
            else:
                r.append(tpl[0])
                r_tags.append(tpl[1])

        new_entity_token = [(i, j) for i, j in zip(r, r_tags)]

        # combine tokens into entities
        flag = False
        entities = []
        ent_tags = []
        for i, tpl in enumerate(new_entity_token):
            if tpl[1] == "O":
                flag = False
                continue
            elif tpl[1] == "I-MISC" or tpl[1] == "I-PER" or tpl[1] == "I-ORG" or tpl[1] == "I-LOC":
                if flag == False:
                    flag = True
                    entities.append(tpl[0])
                    ent_tags.append(tpl[1])
                else:
                    entities[-1] += ' '
                    entities[-1] += tpl[0]
            elif tpl[1] == "B-MISC" or tpl[1] == "B-PER" or tpl[1] == "B-ORG" or tpl[1] == "B-LOC":
                entities.append(tpl[0])
                ent_tags.append(tpl[1])
                
        
        return entities

    def get_response_keywords(self, utterance, topic, entities):
        '''
        this method calls the conv_line model and returns response keywords 
        '''
        entities_comb = ' # '.join(entities)
        input_conv = topic + ' <EOT> '+utterance+' <A0> '+entities_comb+'<A1>'
        '''
        this method calls the conv_line model and returns response keywords 
        '''
        print('input to conv_line')
        print(input_conv)
        np.random.seed(4)
        torch.manual_seed(4)
        maxb = 30 #Can be customized
        minb = 7  #Can be customized
        response = ''
        slines = [input_conv]
        with torch.no_grad():
            #hypotheses = self.conv_line.sample(slines, beam=4, lenpen=2.0, no_repeat_ngram_size=3)
            hypotheses = self.conv_line.sample(slines, sampling=True, sampling_topk=5 ,temperature=0.7 ,lenpen=2.0, max_len_b=maxb, min_len=minb, no_repeat_ngram_size=3)
        hypotheses = hypotheses[0]
        response = hypotheses.replace('\n','')
        keywords = response.replace('<V>','').replace('<s>', '').split('#')
        k = []
        for keyword in keywords:
            keyword = keyword.strip()
            k.append(keyword)
        keywords = k
        return keywords

    def top_k_top_p_filtering(self, logits, filter_value=-float('Inf')):
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
            Args:
                logits: logits distribution shape (batch size x vocabulary size)
                top_k > 0: keep only top k tokens with highest probability (top-k filtering).
                top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                    Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        top_k = min(self.top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if self.top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > self.top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
        return logits

    def sample_sequence(self, model, context):
        context = torch.tensor(context, dtype=torch.long, device=self.device)
        context = context.unsqueeze(0).repeat(1, 1)
        generated = context
        model.cuda()
        with torch.no_grad():
            for _ in trange(self.gen_length):
                inputs = {'input_ids': generated}
                outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
                next_token_logits = outputs[0][:, -1, :] / (self.temperature if self.temperature > 0 else 1.)

                # repetition penalty from CTRL (https://arxiv.org/abs/1909.05858)
                for i in range(1):
                    for _ in set(generated[i].tolist()):
                        next_token_logits[i, _] /= self.repetition_penalty

                filtered_logits = self.top_k_top_p_filtering(next_token_logits)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)
        return generated

    def get_response(self, user_utterance, user_utt_topic, res_keywords):
        '''
        this method calls the dial_gen model and returns generated utterance 
        '''
        res_keywords = ' # '.join(res_keywords)
        input_dial_gen = user_utt_topic.strip() + ' <EOT> ' + user_utterance.strip() +  ' <EOU> ' + res_keywords.strip() + ' <EOK> '
        print('input to dialog generation module')
        print(input_dial_gen)
        if self.gen_model_type =='dialogpt':
            context_tokens = self.gen_tokenizer.encode(input_dial_gen, add_special_tokens=False)
            out = self.sample_sequence(model=self.gen_model, context = context_tokens)
            out = out[:, len(context_tokens):].tolist()
            response = self.gen_tokenizer.decode(out[0], clean_up_tokenization_spaces=True)
            response = response[: response.find('\n') if self.stop_token else None]
        elif self.gen_model_type =='bart':
            np.random.seed(4)
            torch.manual_seed(4)
            maxb = 128 #Can be customized
            minb = 15  #Can be customized
            response = ''
            slines = [input_dial_gen]
            with torch.no_grad():
                hypotheses = self.gen_model.sample(slines, sampling=True, sampling_topk=5 ,temperature=0.7 ,lenpen=2.0, max_len_b=maxb, min_len=minb, no_repeat_ngram_size=3)
            hypotheses = hypotheses[0]
            response = hypotheses.replace('\n','')
        return response



if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Parameters for our Chatbot')
    parser.add_argument('--seed', type=int, default="1000",
                    help='set seed for reproducability')
    parser.add_argument('--gen_model_type', type=str, default="dialogpt",
                    help='generation model type (dialogpt/bart)')
    parser.add_argument('--gen_model_path', type=str, default="./Models/Dial_gen/dialogpt/",
                    help='the path of generation model')
    parser.add_argument('--conv_line_path', type=str, default="Models/conv_line/",
                    help='the path of conv_line model')
    parser.add_argument('--topic_cls_path', type=str, default="./Models/topic_cls/",
                    help='the path of topic classifier')
    parser.add_argument('--label_dir', type=str, default="./Models/topic_cls/",
                    help='the path including labels.csv')
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="temperature of 0 implies greedy sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="primarily useful for CTRL model; in that case, use 1.2")
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument('--stop_token', type=str, default='\n',
                        help="Token at which text generation will be stopped")

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    interaction = Interaction(args)

    print('Hello, Welcome! I am happy to chat with you :)' )
    while True:
        user_utt = input('USER: ')
        if user_utt.lower() == 'exit':
            break
        user_utt_topic = interaction.get_topic(user_utt)
        print('TOPIC: {}'.format(user_utt_topic))
        user_utt_entities = interaction.get_entities(user_utt)
        print('ENTITIES: {}'.format(user_utt_entities))
        res_utt_keywords = interaction.get_response_keywords(user_utt, user_utt_topic, user_utt_entities)
        print('KEYWORDS: {}'.format(res_utt_keywords))
        sys_utt = interaction.get_response(user_utt, user_utt_topic, res_utt_keywords)
        print('CHATBOT: {}'.format(sys_utt))


