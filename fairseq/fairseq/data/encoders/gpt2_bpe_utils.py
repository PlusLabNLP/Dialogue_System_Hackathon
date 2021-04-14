"""
Byte pair encoding utilities from GPT-2.

Original source: https://github.com/openai/gpt-2/blob/master/src/encoder.py
Original license: MIT
"""

from functools import lru_cache
import json


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class Encoder:

    def __init__(self, encoder, bpe_merges, errors='replace'):
        self.encoder = encoder
        self.decoder = {v:k for k,v in self.encoder.items()}
        self.errors = errors # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v:k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        try:
            import regex as re
            self.re = re
        except ImportError:
            raise ImportError('Please install regex with: pip install regex')

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = self.re.compile(r"""'s|'t|'re|'ve|'m4|<A0>|<EOT>| <EOT>|<V>|<A1>|<A2>| <A0>|<P>| <P>|<A1>| <A1>| <A2>| <V>| </s>|</s>|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def bpe(self, token):
        if token in ['<A0>','<A1>','<A2>','<V>','</s>','<P>','<EOT>','\u0120<A0>','\u0120<A1>','\u0120<A2>','\u0120<V>','\u0120</s>','\u0120<P>','\u0120<EOT>']:
# ['<EOT>', '\u0120<EOT>', 'sent01', '\u0120sent01', 'sent02', '\u0120sent02', 'sent03', '\u0120sent03', 'sent04', '\u0120sent04', 'sent05', '\u0120sent05', 'sent06', '\u0120sent06', 'sent07', '\u0120sent07', 'sent08', '\u0120sent08', 'sent09', '\u0120sent09', 'sent10', '\u0120sent10', 'sent11', '\u0120sent11', 'sent12', '\u0120sent12', 'sent13', '\u0120sent13', 'sent14', '\u0120sent14', 'sent15', '\u0120sent15', 'sent16', '\u0120sent16', 'sent17', '\u0120sent17', 'sent18', '\u0120sent18', 'sent19', '\u0120sent19', 'sent20', '\u0120sent20', 'sent21', '\u0120sent21', 'sent22', '\u0120sent22', 'sent23', '\u0120sent23', 'sent24', '\u0120sent24', 'sent25', '\u0120sent25', 'sent26', '\u0120sent26']:
            #print("inside bpe",token)
            return token
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        # print("Text is",text)
        # print("All is" ,self.re.findall(self.pat, text))
        for token in self.re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            # if token in ['<A0>','<A1>','<A2>','<V>','</s>','\u0120<A0>','\u0120<A1>','\u0120<A2>','\u0120<V>','\u0120</s>']:
                # print("Token is |",token,"|")
                # token= token.lstrip()
            #     bpe_tokens.extend(self.encoder[token])
            # else:
            #     print("Token here is ",token)
            #     print(self.bpe(token),self.bpe(token))
            #     for bpe_token in self.bpe(token).split(' '):
            #         print(bpe_token,self.encoder[bpe_token])
            try:
                bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
            except:
                # print("Text was",text)
                print("Token is",token)
                # print("pattern",self.re.findall(self.pat, text))
        #print("bpe tokens is ",self.re.findall(self.pat, text),'\n',bpe_tokens)
        return bpe_tokens

    def decode(self, tokens):
        #print("Token is   ...",tokens)
        arr = []
        for token in tokens:
            arr.append(self.decoder.get(token, token))
            #print(type(self.decoder.get(token, token)),print(self.decoder.get(token, token)))
        text = ''.join([self.decoder.get(token, token) for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text

def get_encoder(encoder_json_path, vocab_bpe_path):
    encoder_json_path = "/lfs1/tuhin/fairseq/encoder.json" #change to "/...YOUR_DIRECTORY.../Dialogue_System_Hackathon/fairseq/encoder.json"
    with open(encoder_json_path, 'r',encoding="utf-8") as f:
        encoder = json.load(f)
    with open(vocab_bpe_path, 'r', encoding="utf-8") as f:
        bpe_data = f.read()
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    return Encoder(
        encoder=encoder,
        bpe_merges=bpe_merges,
    )
