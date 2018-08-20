import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
from common.util import preprocess, create_co_matrix, ppmi

if __name__ == "__main__":
  text = "You say goodbye and I say hello."
  corpus, word_to_id, id_to_word = preprocess(text)
  vocab_size = len(word_to_id)
  C = create_co_matrix(corpus, vocab_size)
  W = ppmi(C)

  # SVD
  U, S, V = np.linalg.svd(W)

  import ipdb; ipdb.set_trace()