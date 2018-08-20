import sys
sys.path.append("..")
import numpy as np
from common.util import preprocess, create_co_matrix, cos_similarity, most_similar, ppmi

if __name__ == "__main__":
  text = "You say goodbye and I say hello."
  corpus, word_to_id, id_to_word = preprocess(text)
  vocab_size = len(word_to_id)
  C = create_co_matrix(corpus, vocab_size)

  c0 = C[word_to_id['i']]
  c1 = C[word_to_id['you']]

  # print(cos_similarity(c0, c1))
  # most_similar('you', word_to_id, id_to_word, C)
  W = ppmi(C)

  np.set_printoptions(precision=3)
  print('Convariance Matrix')
  print(C)
  print("-"*50)
  print("PPMI")
  print(W)

