import tensorflow as tf
import numpy as np
from copy import deepcopy
class BeamPath():
    def __init__(self, tokenizer_wrapper, max_sentence_length, sentence_tokens=[], hidden=None, prob=[]):
        self.sentence_tokens=sentence_tokens
        self.hidden = hidden
        self.prob = prob
        self.tokenizer_wrapper=tokenizer_wrapper
        self.max_sentence_length = max_sentence_length

    def ended(self):
        if len(self.sentence_tokens) > 0:
            return len(self.sentence_tokens) >= self.max_sentence_length or self.tokenizer_wrapper.get_word_from_token(self.sentence_tokens[-1]) == 'endseq'
        return False

    def get_sentence_words(self):
        words = []
        for token in self.sentence_tokens:
            word = self.tokenizer_wrapper.get_word_from_token(token)
            if word != 'endseq':
                words.append(word)
        return words

    def __deepcopy__(self, memodict={}):
        copy = BeamPath(self.tokenizer_wrapper,self.max_sentence_length, deepcopy(self.sentence_tokens),deepcopy(self.hidden),deepcopy(self.prob))
        return copy

    def add_token(self,token):
        self.sentence_tokens.append(token)

    def add_probability(self,prob):
        self.prob.append(prob)

    def add_record(self,token, prob, hidden):
        self.sentence_tokens.append(token)
        self.prob.append(prob)
        self.hidden = hidden

    def get_input_tensor(self):
        return tf.expand_dims([self.sentence_tokens[-1]], 0)

    def get_last_token(self):
        return self.sentence_tokens[-1]

    def get_total_probability(self):
        return np.sum(-np.log(self.prob))

    def get_hidden_size(self):
        return self.hidden.shape

    def get_hidden_layer(self):
        return self.hidden



