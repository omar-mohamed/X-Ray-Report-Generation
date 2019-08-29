
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from nltk.translate import bleu_score,meteor_score
import numpy as np
x=[1,2,2,3]
hyp=['am','leg']
ref=['am','legend']
# print(bleu_score.sentence_bleu([ref],hyp,weights=[0.5,0.5]))
print(bleu_score.corpus_bleu([[ref]],[hyp],weights=[1.0]))

# plt.imsave(path)

# import tensorflow as tf

# print(tf.__version__)

# conda config --add channels conda-forge
# conda install keras opencv shapely tensorflow gensim pandas imgaug
# pip install --upgrade tensorflow==2.0.0-beta1
# pip install  "C:\Users\omarm\Downloads\Shapely-1.6.4.post2-cp37-cp37m-win_amd64.whl" matplotlib  pandas imgaug gensim tensorflow==2.0.0-beta1