import pandas as pd
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import text_to_word_sequence
import numpy as np

class TokenizerWrapper:
    def __init__(self,dataset_csv_file,class_name,max_caption_length):
        dataset_df = pd.read_csv(dataset_csv_file)
        sentences = dataset_df[class_name].tolist()
        self.max_caption_length=max_caption_length
        self.init_tokenizer(sentences)

    def clean_sentence(self,sentence):
        return text_to_word_sequence(sentence, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')

    def init_tokenizer(self,sentences):

        for i in range(len(sentences)):
            if pd.isna(sentences[i]):
                sentences[i] = ""
            sentences[i] = self.clean_sentence(sentences[i])

        # Tokenize the reviews
        print("Tokenizing dataset..")
        self.tokenizer = Tokenizer(oov_token='UNK')
        self.tokenizer.fit_on_texts(sentences)  # give each word a unique id
        print("number of tokens: {}".format(self.tokenizer.word_index))
        print("Tokenizing is complete.")

    def get_tokenizer_word_index(self):
        return len(self.tokenizer.word_index)+1

    def get_token_of_word(self,word):
        return self.tokenizer.word_index[word]

    def get_word_tokens_list(self):
        return self.tokenizer.word_index


    def tokenize_sentences(self,sentences):
        index = 0
        tokenized_sentences = np.zeros((sentences.shape[0], self.max_caption_length), dtype=int)
        for caption in sentences:
            tokenized_caption = self.tokenizer.texts_to_sequences([self.clean_sentence(caption[0])])
            tokenized_sentences[index] = pad_sequences(tokenized_caption, maxlen=self.max_caption_length,
                                          padding='post')  # padded with max length
            index = index + 1
        return tokenized_sentences