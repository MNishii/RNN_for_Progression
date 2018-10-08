# -*- coding: utf-8 -*- 

import numpy as np
from functions import *
import matplotlib.pyplot as plt
import pandas as pd
import pickle

##### モデル設定 #####
input_layer_size = 8
hidden_layer_size = 100
output_layer_size = input_layer_size

#学習率設定
learning_rate = 0.1

iteration = 20000
plot_interval = 500

# 重み行列とバイアスの設定
params = {}
params["W1"] = np.random.randn(input_layer_size, hidden_layer_size)/np.sqrt(input_layer_size)
params["b1"] = np.random.randn(hidden_layer_size)
params["W2"] = np.random.randn(hidden_layer_size, output_layer_size)/np.sqrt(hidden_layer_size)
params["b2"] = np.random.randn(output_layer_size)

#順伝播
def forward(params, x):
    W1, W2 = params["W1"], params["W2"]
    b1, b2 = params["b1"], params["b2"]
    f1 = np.dot(x, W1) + b1
    z1 = relu(f1)
    f2 = np.dot(z1, W2) + b2
    y  = sigmoid(f2)
    return z1, y

#逆伝播
def backward(params, x, z1, y):
    grad = {}
    W1, W2 = params["W1"], params["W2"]
    b1, b2 = params["b1"], params["b2"]
    delta2 = d_least_square(y, y_sample)*(1-y)*y
    grad["b2"] = np.sum(delta2, axis = 0)   
    grad["W2"] = np.dot(z1.reshape(-1, 1), delta2) 
    delta1 = np.dot(delta2, W2.T)               
    delta1_r = delta1*d_relu(z1)  
    grad["b1"] = np.sum(delta1_r, axis = 0) 
    grad["W1"] = np.dot(x.reshape(-1, 1), delta1_r)
    return grad

#結果データ格納リスト
accuracy_list = []


##### 教師データ作成 #####
x_train = []
y_train = []

for j in range(50000):
    data = np.random.rand(input_layer_size)
    _, val = random_seq(data)
    x_train.append(data)
    y_train.append(val)

x_train = np.array(x_train)
y_train = np.array(y_train)

##### トレーニング #####
for i in range(iteration):
    choice = np.random.choice(len(x_train), 1) # バッチサイズ1で逐次的にs処理
    x_sample = x_train[int(choice)]
    y_sample = y_train[int(choice)]

    z1, y = forward(params, x_sample)
    y_sample = y_sample.reshape(1, -1)     # ベクトル形式を(1,)から(1, 1)へ再定義
    loss = least_square(y_sample, y)       # ロス関数として、誤差二乗平均を使用
    grad = backward(params, x_sample, z1, y)

    #出力を 0 or 1 に変換する
    for j in range(len(y)):
        if y[j] >= 0.5:
            y[j] = 1
        else:
            y[j] = 0

    # 勾配の更新
    for key in ("W1", "W2", "b1", "b2"):
        params[key] -= learning_rate * grad[key]
    
    ##### 正答率算出 #####
    count = 0
    if(i % plot_interval == 1):
        print("iters:" + str(i))
        acc = 0
        for k in range(len(y)):
            acc += np.sum(y[k] == y_sample[0,k])
        print('y', y)
        print('y_sample', y_sample[0,:])
        print('accuracy', acc)
        accuracy_list.append(acc)
        count = 0

# 正答率の保存
accuracy_list = pd.DataFrame(accuracy_list)
accuracy_list.to_csv('accuracy_list_dnn.csv')

# パラメータの保存
file_name="parameter.pkl"
with open(file_name, 'wb') as f:
    pickle.dump(params, f)
print("Saved Parameters!")