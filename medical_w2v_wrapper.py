import gensim
import numpy as np

class Medical_W2V_Wrapper:
    def __init__(self):
        self.word_embeddings = gensim.models.KeyedVectors.load_word2vec_format("medical_word_embeddings/pubmed2018_w2v_400D.bin",
                                                              binary=True)



    def get_embeddings_matrix_for_words(self,word_tokens,vocab_size):
        embeddings=np.random.uniform(low=1e-5,high=1,size=(vocab_size,self.word_embeddings['the'].shape[0]))
        word_counter=0
        for word, token in word_tokens.items():
            try:
                embeddings[token,:]=self.word_embeddings[word]
            except:
                print("Word: {} not found in medical word embeddings".format(word))
            word_counter+=1
            if word_counter == vocab_size:
                break
        return embeddings

