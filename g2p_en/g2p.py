# -*- coding: utf-8 -*-
# /usr/bin/python
'''
By kyubyong park(kbpark.linguist@gmail.com) and Jongseok Kim(https://github.com/ozmig77)
https://www.github.com/kyubyong/g2p
'''
from nltk import pos_tag
from nltk.corpus import cmudict
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import codecs
import re
import os
import unicodedata
from builtins import str as unicode
from .expand import normalize_numbers

try:
    nltk.data.find('taggers/averaged_perceptron_tagger.zip')
except LookupError:
    nltk.download('averaged_perceptron_tagger')
try:
    nltk.data.find('corpora/cmudict.zip')
except LookupError:
    nltk.download('cmudict')

dirname = os.path.dirname(__file__)

def construct_homograph_dictionary(f):
    homograph2features = dict()
    for line in codecs.open(f, 'r', 'utf8').read().splitlines():
        if len(line)==0: continue # blank
        if line.startswith("#"): continue # comment
        headword, pron1, pron2, pos1 = line.strip().split("|")
        homograph2features[headword.lower()] = (pron1.split(), pron2.split(), pos1)
    return homograph2features

def kaldi_tokenize(raw_sentence):
  seq, prev = [],0
  for m in re.finditer(r'(\w|\’\w|\'\w)+', raw_sentence, re.UNICODE):
    start, end = m.span()
    spacer = raw_sentence[prev:start].strip()
    if len(spacer)>0:
      seq.append( spacer )
    word = m.group()
    #token = kaldi_normalize(word, self.vocab)
    token = word.lower().replace("’", "'")
    seq.append( token )
    prev=end
  spacer = raw_sentence[prev:].strip()
  if len(spacer)>0:
    seq.append( spacer )
  return seq
  


# def segment(text):
#     '''
#     Splits text into `tokens`.
#     :param text: A string.
#     :return: A list of tokens (string).
#     '''
#     print(text)
#     text = re.sub('([.,?!]( |$))', r' \1', text)
#     print(text)
#     return text.split()

class G2p(object):
    def __init__(self):
        super().__init__()
        self.graphemes = ["<pad>", "<unk>", "</s>"] + list("abcdefghijklmnopqrstuvwxyz")
        self.phonemes = ["<pad>", "<unk>", "<s>", "</s>"] + ['AA0', 'AA1', 'AA2', 
                                                             'AE0', 'AE1', 'AE2', 
                                                             'AH0', 'AH1', 'AH2', 
                                                             'AO0', 'AO1', 'AO2', 
                                                             'AW0', 'AW1', 'AW2', 
                                                             'AY0', 'AY1', 'AY2', 
                                                             'B', 'CH', 'D', 'DH',
                                                             'EH0', 'EH1', 'EH2', 
                                                             'ER0', 'ER1', 'ER2', 
                                                             'EY0', 'EY1', 'EY2', 
                                                             'F', 'G', 'HH',
                                                             'IH0', 'IH1', 'IH2', 
                                                             'IY0', 'IY1', 'IY2', 
                                                             'JH', 'K', 'L',
                                                             'M', 'N', 'NG', 
                                                             'OW0', 'OW1', 'OW2', 
                                                             'OY0', 'OY1', 'OY2', 
                                                             'P', 'R', 'S', 'SH', 'T', 'TH',
                                                             'UH0', 'UH1', 'UH2', 
                                                             'UW',
                                                             'UW0', 'UW1', 'UW2', 
                                                             'V', 'W', 'Y', 'Z', 'ZH']
                                                             
        self.tokens = self.phonemes + ['<">', '<->', '<,>', '<.>', '<;>', '< >', ]
                                                             
        self.g2idx = {g: idx for idx, g in enumerate(self.graphemes)}
        self.idx2g = {idx: g for idx, g in enumerate(self.graphemes)}

        self.p2idx = {p: idx for idx, p in enumerate(self.phonemes)}
        self.idx2p = {idx: p for idx, p in enumerate(self.phonemes)}

        self.tok2idx = {t: idx for idx, t in enumerate(self.tokens)}
        self.idx2tok = {idx: t for idx, t in enumerate(self.tokens)}

        self.cmu = cmudict.dict()
        self.load_variables()
        
        self.homograph2features = construct_homograph_dictionary(os.path.join(dirname, 'homographs.en'))
        corrections = construct_homograph_dictionary(os.path.join(dirname, '..', '..', 'gentle-fixes.en'))
        self.homograph2features.update( corrections )

    def load_variables(self):
        self.variables = np.load(os.path.join(dirname,'checkpoint20.npz'))
        self.enc_emb = self.variables["enc_emb"]  # (29, 64). (len(graphemes), emb)
        self.enc_w_ih = self.variables["enc_w_ih"]  # (3*128, 64)
        self.enc_w_hh = self.variables["enc_w_hh"]  # (3*128, 128)
        self.enc_b_ih = self.variables["enc_b_ih"]  # (3*128,)
        self.enc_b_hh = self.variables["enc_b_hh"]  # (3*128,)

        self.dec_emb = self.variables["dec_emb"]  # (74, 64). (len(phonemes), emb)
        self.dec_w_ih = self.variables["dec_w_ih"]  # (3*128, 64)
        self.dec_w_hh = self.variables["dec_w_hh"]  # (3*128, 128)
        self.dec_b_ih = self.variables["dec_b_ih"]  # (3*128,)
        self.dec_b_hh = self.variables["dec_b_hh"]  # (3*128,)
        self.fc_w = self.variables["fc_w"]  # (74, 128)
        self.fc_b = self.variables["fc_b"]  # (74,)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def grucell(self, x, h, w_ih, w_hh, b_ih, b_hh):
        rzn_ih = np.matmul(x, w_ih.T) + b_ih
        rzn_hh = np.matmul(h, w_hh.T) + b_hh

        rz_ih, n_ih = rzn_ih[:, :rzn_ih.shape[-1] * 2 // 3], rzn_ih[:, rzn_ih.shape[-1] * 2 // 3:]
        rz_hh, n_hh = rzn_hh[:, :rzn_hh.shape[-1] * 2 // 3], rzn_hh[:, rzn_hh.shape[-1] * 2 // 3:]

        rz = self.sigmoid(rz_ih + rz_hh)
        r, z = np.split(rz, 2, -1)

        n = np.tanh(n_ih + r * n_hh)
        h = (1 - z) * n + z * h

        return h

    def gru(self, x, steps, w_ih, w_hh, b_ih, b_hh, h0=None):
        if h0 is None:
            h0 = np.zeros((x.shape[0], w_hh.shape[1]), np.float32)
        h = h0  # initial hidden state
        outputs = np.zeros((x.shape[0], steps, w_hh.shape[1]), np.float32)
        for t in range(steps):
            h = self.grucell(x[:, t, :], h, w_ih, w_hh, b_ih, b_hh)  # (b, h)
            outputs[:, t, ::] = h
        return outputs

    def encode(self, word):
        chars = list(word) + ["</s>"]
        x = [self.g2idx.get(char, self.g2idx["<unk>"]) for char in chars]
        x = np.take(self.enc_emb, np.expand_dims(x, 0), axis=0)
        return x

    def predict(self, word):
        # encoder
        enc = self.encode(word)
        enc = self.gru(enc, len(word) + 1, self.enc_w_ih, self.enc_w_hh,
                       self.enc_b_ih, self.enc_b_hh, h0=np.zeros((1, self.enc_w_hh.shape[-1]), np.float32))
        last_hidden = enc[:, -1, :]

        # decoder
        dec = np.take(self.dec_emb, [2], axis=0)  # 2: <s>
        h = last_hidden

        preds = []
        for i in range(20):
            h = self.grucell(dec, h, self.dec_w_ih, self.dec_w_hh, self.dec_b_ih, self.dec_b_hh)  # (b, h)
            logits = np.matmul(h, self.fc_w.T) + self.fc_b
            pred = logits.argmax()
            if pred == 3: break  # 3: </s>
            preds.append(pred)
            dec = np.take(self.dec_emb, [pred], axis=0)

        preds = [self.idx2p.get(idx, "<unk>") for idx in preds]
        return preds

    def __call__(self, text):
        # preprocessing
        text = unicode(text)
        text = normalize_numbers(text)
        text = ''.join(char for char in unicodedata.normalize('NFD', text)
                       if unicodedata.category(char) != 'Mn')  # Strip accents
        text = text.lower()
        #text = re.sub("[^ a-z'.,?!\-]", "", text)
        text = re.sub("[^ a-z'.,?!\-;:\"]", "", text)   # mdda
        #text = re.sub("([a-z])\-([a-z])", r"\1 - \2", text)   # mdda 'hot-shot' -> 'hot - shot'
        text = re.sub("([a-z])\-([a-z])", r"\1 \2", text)   # mdda    'hot-shot' -> 'hot shot'
        text = text.replace("i.e.", "that is")
        text = text.replace("e.g.", "for example")

        # tokenization
        #words1 = word_tokenize(text)
        #print( words1 )
        words2 = kaldi_tokenize(text)
        #print( words2 )
        
        tokens = pos_tag(words2)  # tuples of (word, tag)

        # steps
        prons = []
        for word, pos in tokens:
            if re.search("[a-z]", word) is None:
                pron = [word]

            elif word in self.homograph2features:  # Check homograph
                pron1, pron2, pos1 = self.homograph2features[word]
                if pos.startswith(pos1):
                    pron = pron1
                else:
                    pron = pron2
            elif word in self.cmu:  # lookup CMU dict
                pron = self.cmu[word][0]
            else: # predict for oov
                pron = self.predict(word)

            #prons.extend(pron)  #mdda
            #prons.extend([" "]) #mdda
            prons.append( (word, pron) )   #mdda

        #return prons[:-1]   #mdda
        return prons         #mdda

    def silence_to_token(self, span):
      #print(f"span='{span:s}' : {start:d}-{end:d} '{t:s}'")
      #print(f"span='{span:s}'")
      for special in '"-,.;':  # This returns early depending on first symbol found
        if special in span:
          return '<'+special+'>'
      return '< >'


if __name__ == '__main__':
    texts = ["I have $250 in my pocket.", # number -> spell-out
             "popular pets, e.g. cats and dogs", # e.g. -> for example
             "I refuse to collect the refuse around here.", # homograph
             "I'm an activationist."] # newly coined word
    g2p = G2p()
    
