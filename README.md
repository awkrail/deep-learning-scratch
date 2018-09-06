# Deep Learning Scratch

# ch01
ニューラルネットの基本的な仕組みと実装について

## 疑問, 試したいこと
- SoftmaxとCrossEntropyのレイヤでのbackward()時の処理について
- 全体を通してどういう風にbackward()で誤差が逆伝搬しているのか, ソースコードで追ってみる
- plt.ylim(*ylim) <= なにこれ?
- emove_duplicate()関数とclip_grads()関数について
- Trainerを使って学習させてみる

# ch02
単語の分散表現 : word2vec以前

## 疑問, 試したいこと
- シソーラスとWordNetで遊ぶ
- 相互情報量
  - どれくらい不確実性が減るのかを表現する。 : 'the'はよくでてくるものであり, 不確実性があまり減らない。
  - 逆に頻度が小さいものがでてくると, 次に共起するものがだいたいわかるので, 不確実性が減少する
- SVDの理論的な背景

# ch03
単語の分散表現 : CBoW

## 疑問, 試したいこと
- CBoWをはじめとして分散表現の論文を読む
- convert_one_hot()関数は実装を読んでない
- Adamの実装も読みたい
- skip-gramの実装を読む

# ch04
分散表現②(分散表現の高速化) : Skip-gram, Negative-Sampling

## 疑問, 試したいこと
- word2vecのEmbeddingレイヤで重複を取り除くためになぜ代入ではなく加算を行うのだろうか
  - 代入ではなく足し算するのはバッチを処理する時に1つのデータの結果ではなく複数の結果を反映させるため
- EmbeddingDotクラスでは正解のぶぶんのtargetWとh(隠れ層のベクトル)との内積をとるときに重複について処理は不要なのか
  - => 処理してましたすいません(common/layer.pyのEmbeddingクラスのbackwardをみた)