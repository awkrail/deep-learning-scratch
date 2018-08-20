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