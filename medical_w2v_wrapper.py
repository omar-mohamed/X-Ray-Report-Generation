import gensim
import numpy as np
import pickle
import os

class Medical_W2V_Wrapper:
    def __init__(self):
        if os.path.isfile("medical_word_embeddings/saved_embeddings.pickle"):
            with open('medical_word_embeddings/saved_embeddings.pickle', 'rb') as handle:
                self.word_embeddings = pickle.load(handle)
        else:
            self.word_embeddings = gensim.models.KeyedVectors.load_word2vec_format(
                "medical_word_embeddings/pubmed2018_w2v_400D.bin",
                binary=True)

    def get_embeddings_matrix_for_words(self, word_tokens, vocab_size):
        embeddings = np.zeros(shape=(vocab_size, self.word_embeddings['the'].shape[0]))
        word_counter = 0
        for word, token in word_tokens.items():
            try:
                embeddings[token, :] = self.word_embeddings[word]
            except:
                print("Word: {} not found in medical word embeddings".format(word))
            word_counter += 1
            if word_counter == vocab_size:
                break
        return embeddings

    def get_embeddings_matrix_for_tags(self, tag_classes):
        embeddings = np.zeros(shape=(len(tag_classes), self.word_embeddings['the'].shape[0]))
        token = 0
        for _class in tag_classes:
            if _class in self.word_embeddings:
                embeddings[token, :] = self.word_embeddings[_class]
            else:
                sentence = _class.split()
                sentence_vec = np.zeros(self.word_embeddings['the'].shape[0])
                for word in sentence:
                    sentence_vec += self.word_embeddings[word]

                embeddings[token, :] = sentence_vec

            token += 1
        return embeddings

    def save_embeddings(self, word_tokens):
        word_counter = 0
        dictionary={}
        for word, token in word_tokens.items():
            try:
                dictionary[word] = self.word_embeddings[word]
            except:
                dictionary[word] = np.zeros(shape= self.word_embeddings['the'].shape[0])

            word_counter += 1
        print("saved {} words".format(word_counter))
        with open('medical_word_embeddings/saved_embeddings.pickle', 'wb') as handle:
            pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
