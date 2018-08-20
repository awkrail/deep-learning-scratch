import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
from common.util import preprocess, create_co_matrix, ppmi, most_similar
from dataset import ptb
from sklearn.utils.extmath import randomized_svd

if __name__ == "__main__":
  window_size = 2
  wordvec_size = 100

  corpus, word_to_id, id_to_word = ptb.load_data('train')
  vocab_size = len(word_to_id)
  C = create_co_matrix(corpus, vocab_size)
  W = ppmi(C)

  # SVD
  U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=None)

  word_vecs = U[:, :wordvec_size]

  querys = ['you', 'year', 'car', 'toyota']
  for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)