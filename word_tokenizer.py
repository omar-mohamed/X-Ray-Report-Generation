import nltk
import re
import math
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences



def tokenize_sentence(sentence):
    bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '',
                                t.replace('"', '').replace('/', '').replace('\\', '').replace("'",

                                                                                              '').strip().lower()).split()
    if pd.isna(sentence[i]):
        sentence[i]=""
    sentence[i]=bioclean(sentence[i])


data=pd.read_csv("IU-XRay/all_data.csv")
findings=data["Findings"].tolist()
Image_Indexes=data["Image Index"].tolist()
# sentence="who is this mad man"

# nltk_tokens = nltk.word_tokenize(sentence)
# print (nltk_tokens)

bioclean = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '',
                            t.replace('"', '').replace('/', '').replace('\\', '').replace("'",
                                                                                          '').strip().lower()).split()
for i in range(len(findings)):
    if pd.isna(findings[i]):
        findings[i]= ""
    findings[i]=bioclean(findings[i])

# Tokenize the reviews
print("Tokenizing dataset..")
tokenizer = Tokenizer()
tokenizer.fit_on_texts(findings)  # give each word a unique id
print("Tokenizing is complete.")

word_index = tokenizer.word_index  # final id
print("word index: " + str(word_index))

train_seq = tokenizer.texts_to_sequences(findings)  # convert dataset to ids
print("train_seq is complete.")

# Pad the reviews

max_review_length = 170

print("Padding dataset..")
train_pad = pad_sequences(train_seq, maxlen=max_review_length,padding='post')  # padded with max length
review_lengths_longer_than_pad = 0
for seq in train_seq:  # calculate how many reviews longer than pad length
    if len(seq) > max_review_length:
        review_lengths_longer_than_pad = review_lengths_longer_than_pad + 1

print("Number of reviews longer than pad length({}): {}".format(max_review_length, review_lengths_longer_than_pad))

print("train_pad is complete.")