import torch
from transformers import AutoModelWithLMHead, AutoTokenizer
from fairseq.models.bart import BARTModel
import numpy as np




class Interaction():

    def __init__(self, args):
        self.gen_model_type = args.gen_model_type
        self.gen_model_path = args.gen_model_path
        self.gen_length = args.length
        self.temperature = args.temperature
        self.top_k = args.top_k
        self.top_p = args.top_p
        self.repetition_penalty= args.repetition_penalty
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.gen_model_type = gen_model_type.lower()

        if self.gen_model_type =='dialogpt':
            self.tokenizer = AutoTokenizer.from_pretrained(self.gen_model_path)
            self.gen_model = AutoModelWithLMHead.from_pretrained(self.gen_model_path)
            self.gen_model.cuda()
            self.gen_model.eval()
        else:
            self.gen_model = BARTModel.from_pretrained(self.gen_model_path,checkpoint_file='checkpoint_best.pt',data_name_or_path=self.gen_model_path)
            self.gen_model.cuda()
            self.gen_model.eval()

        

    def get_topic(self, utterance):
        '''
        this method calls the topic cls and returns utterace's topic
        '''
        topic =''
        return topic

    def get_entities(self, utterance):
        '''
        this method calls the entity extractor model and returns utterace's entities
        '''
        #Zhubo --> entity extractor
        entities =''
        return entities

    def get_response_keywords(self, utterance, topic, entities):
        '''
        this method calls the conv_line model and returns response keywords 
        '''
        #Tuhin --> conv_line
        keywords =''
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

    def sample_sequence(self, model, context, is_xlnet=False, is_xlm_mlm=False, xlm_mask_token=None, xlm_lang=None):
        context = torch.tensor(context, dtype=torch.long, device=self.device)
        context = context.unsqueeze(0).repeat(1, 1)
        generated = context
        model.cuda()
        with torch.no_grad():
            for _ in trange(length):
                inputs = {'input_ids': generated}
                outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
                next_token_logits = outputs[0][:, -1, :] / (self.temperature if self.temperature > 0 else 1.)

                # repetition penalty from CTRL (https://arxiv.org/abs/1909.05858)
                for i in range(1):
                    for _ in set(generated[i].tolist()):
                        next_token_logits[i, _] /= self.repetition_penalty

                filtered_logits = self.top_k_top_p_filtering(next_token_logits, top_k=self.top_k, top_p=self.top_p)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)
        return generated

    def get_response(self, user_utterance, user_utt_topic, res_keywords):
        '''
        this method calls the dial_gen model and returns generated utterance 
        '''
        if self.gen_model_type == 'dialogpt':
            input_dial_gen = user_utt_topic.strip() + ' <EOT> ' + user_utterance.strip() +  ' <EOU> ' + res_keywords.strip() + ' <EOK> '
        else:
            input_dial_gen = user_utt_topic.strip() + ' <EOT> ' + user_utterance.strip() +  ' <V> ' + res_keywords.strip()


        if self.gen_model_type =='dialogpt':
            context_tokens = self.tokenizer.encode(input_dial_gen, add_special_tokens=False)
            out = self.sample_sequence(
                    model=self.gen_model,
                    context=context_tokens,
                    num_samples=1,
                    length=self.gen_length,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    repetition_penalty=self.repetition_penalty,
                    is_xlnet=False,
                    is_xlm_mlm=False,
                    xlm_mask_token=None,
                    xlm_lang=None,
                    device=self.device,
                )
            out = out[:, len(context_tokens):].tolist()
            response = self.tokenizer.decode(out[0], clean_up_tokenization_spaces=True)
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
    parser.add_argument('--gen_model_path', type=str, default="./Models/Dial_gen/dialogpt",
                    help='the path of generation model')
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
        user_utt = input('User: ')
        user_utt_topic = interaction.get_topic(user_utt)
        user_utt_entities = interaction.get_entities(user_utt)
        res_utt_keywords = interaction.get_response_keywords(user_utt, user_utt_topic, user_utt_entities)
        sys_utt = interaction.get_response(user_utt, user_utt_topic, res_utt_keywords)
        print('Chatbot: {}'.format(sys_utt))


