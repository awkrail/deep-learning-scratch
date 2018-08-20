import sys
sys.path.append("..")
from dataset import ptb

if __name__ == "__main__":
  corpus, word_to_id, id_to_word = ptb.load_data('train')

  print("corpus size : ", len(corpus))
  print("corpus[:30] : ", corpus[:30])
  print()
  print("id_to_word[0] : ", id_to_word[0])
  print("id_to_word[1] : ", id_to_word[1])