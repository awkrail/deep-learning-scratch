import sys
sys.path.append("..")
import numpy as np
from common.layers import Affine, Sigmoid, SoftmaxWithLoss
from common.optimizer import SGD
from dataset import spiral
import matplotlib.pyplot as plt


class TwoLayerNet:
  def __init__(self, input_size, hidden_size, output_size):
    I, H, O = input_size, hidden_size, output_size

    # 重みのバイアスの初期化
    W1 = np.random.randn(I, H)
    b1 = np.zeros(H)
    W2 = np.random.randn(H, O)
    b2 = np.zeros(O)

    # レイヤの生成
    self.layers = [
      Affine(W1, b1),
      Sigmoid(),
      Affine(W2, b2)
    ]
    self.loss_layer = SoftmaxWithLoss()

    # 全ての重みと勾配をリストにまとめる
    self.params, self.grads = [], []
    for layer in self.layers:
      self.params += layer.params
      self.grads += layer.grads
  
  def predict(self, x):
    for layer in self.layers:
      x = layer.forward(x)
    return x
  
  def forward(self, x, t):
    y = self.predict(x)
    loss = self.loss_layer.forward(y, t)
    return loss
  
  def backward(self, dout=1):
    dout = self.loss_layer.backward(dout)
    for layer in reversed(self.layers):
      dout = layer.backward(dout)
    return dout

if __name__ == "__main__":
  max_epoch = 300
  batch_size = 30
  hidden_size = 10
  learning_rate = 1.0

  x, t = spiral.load_data()
  model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
  optimizer = SGD(lr=learning_rate)

  # 学習に使用する変数
  data_size = len(x)
  max_iters = data_size // batch_size
  total_loss = 0
  loss_count = 0
  loss_list = []

  for epoch in range(max_epoch):
    # データシャッフル
    idx = np.random.permutation(data_size)
    x = x[idx]
    t = t[idx]

    for iters in range(max_iters):
      batch_x = x[iters*batch_size:(iters+1)*batch_size]
      batch_t = t[iters*batch_size:(iters+1)*batch_size]
      
      # 勾配を計算する
      loss = model.forward(batch_x, batch_t)
      model.backward()
      optimizer.update(model.params, model.grads)

      total_loss += loss
      loss_count += 1

      if (iters+1) % 10 == 0:
        avg_loss = total_loss / loss_count
        print("epoch %d : iter %d / %d : loss %.2f" %(epoch+1, iters+1, max_iters, avg_loss))
        loss_list.append(avg_loss)
        total_loss, loss_count = 0, 0