# -*- coding: utf-8 -*- 

import numpy as np
from functions import *
import matplotlib.pyplot as plt
import pandas as pd
import pickle

##### モデル設定 #####
sequence_length = 8
input_layer_size = 1
hidden_layer_size = 100
output_layer_size = 1

#学習率設定
learning_rate = 0.1

iters_num = 20000
plot_interval = 500

# 重み行列とバイアスの設定
params = {}
params['W_in'] = np.random.randn(input_layer_size, hidden_layer_size)/(np.sqrt(input_layer_size))
params['W_out'] = np.random.randn(hidden_layer_size, output_layer_size)/(np.sqrt(hidden_layer_size))
params['W'] = np.random.randn(hidden_layer_size, hidden_layer_size)/(np.sqrt(hidden_layer_size))
params['b'] = np.zeros(hidden_layer_size)
params['b_out'] = np.zeros(output_layer_size)

# 勾配の設定
W_in_grad = np.zeros_like(params['W_in'])
W_out_grad = np.zeros_like(params['W_out'])
W_grad = np.zeros_like(params['W'])
W_b_grad = np.zeros_like(params['b'])
W_b_out_grad = np.zeros_like(params['b_out'])

# 各時系列におけるデータの格納
u = np.zeros((hidden_layer_size, sequence_length)) 
z = np.zeros((hidden_layer_size, sequence_length + 1)) # 回帰的な内部状態。BPTTが計算できるよう、1を足しておく。
y_ = np.zeros((output_layer_size, sequence_length))
y = np.zeros((output_layer_size, sequence_length))

delta_out = np.zeros((output_layer_size, sequence_length))
delta_out_2 = np.zeros((hidden_layer_size, sequence_length + 1)) 
delta_ = np.zeros((hidden_layer_size, sequence_length)) 
delta = np.zeros((hidden_layer_size, sequence_length + 1)) # z と同様に BPTTが計算できるよう、1を足しておく。

#結果データ格納リスト
accuracy_list = []

##### 教師データ作成 #####
x_train = []
y_train = []

for j in range(50000):
    data = np.random.rand(sequence_length)
    _, val = random_seq(data)
    x_train.append(data)
    y_train.append(val)

x_train = np.array(x_train)
y_train = np.array(y_train)

##### トレーニング #####
for i in range(iters_num):
    choice = np.random.choice(len(x_train), 1) # バッチサイズ1で逐次的に処理
    x_sample = x_train[int(choice)]
    y_sample = y_train[int(choice)]

    # 時系列ループ
    for t in range(sequence_length):

        u[:,t] = np.dot(x_sample[t].reshape(1, -1), params['W_in']) + np.dot(z[:,t].reshape(1, -1), params['W']) + params['b']
        z[:,t+1] = np.tanh(u[:,t])
        y_[:,t] = np.dot(z[:,t+1].reshape(1, -1), params['W_out']) + params['b_out']
        y[:,t] = sigmoid(y_[:,t])
        loss = least_square(y[:,t], y_sample[t])

    #Backward
    for t in range(sequence_length)[::-1]:
        delta_out[:,t] = (y[:,t] - y_sample[t]) * (1-y[:,t])*y[:,t]
        delta_out_2[:,t+1] = np.dot(delta_out[:,t].reshape(1,-1), params['W_out'].T)
        delta_[:,t] = (delta_out_2[:,t+1] + delta[:, t+1]) * d_tanh(u[:,t])
        delta[:,t] = np.dot(delta_[:,t], params['W'].T)

        # 各 t において勾配の値を蓄積する
        W_out_grad += np.dot(z[:,t+1].reshape(-1,1), delta_out[:,t].reshape(1,-1))
        W_b_out_grad += np.sum(delta_out[:,t].reshape(1,-1), axis = 0)
        W_grad += np.dot(z[:,t].reshape(-1,1), delta_[:,t].reshape(1,-1))
        W_in_grad += np.dot(x_sample[t].reshape(-1, 1), delta_[:,t].reshape(1,-1))
        W_b_grad += np.sum(delta_[:,t].reshape(1,-1), axis = 0)

    #出力を 0 or 1 に変換する
    for j in range(len(y[0])):
        if y[:,j] >= 0.5:
            y[:,j] = 1
        else:
            y[:,j] = 0

    # 勾配の更新
    params['W_in'] -= learning_rate * W_in_grad
    params['W_out'] -= learning_rate * W_out_grad
    params['W'] -= 0.001 * W_grad
    params['b_out'] -= learning_rate * W_b_out_grad
    params['b'] -= 0.001 * W_b_grad
        
    W_in_grad *= 0
    W_out_grad *= 0
    W_grad *= 0
    W_b_out_grad *= 0
    W_b_grad *= 0

    ##### 正答率算出 #####
    count = 0
    if(i % plot_interval == 1):
        print("iters:" + str(i))
        acc = 0
        for k in range(len(y[0,:])):
            acc += np.sum(y[0,k] == y_sample[k])
        print('y', y[0,:])
        print('y_sample', y_sample)
        print('accuracy', acc)
        accuracy_list.append(acc)
        count = 0

# 正答率の保存
accuracy_list = pd.DataFrame(accuracy_list)
accuracy_list.to_csv('accuracy_list_rnn.csv')

# パラメータの保存
file_name="parameter.pkl"
with open(file_name, 'wb') as f:
    pickle.dump(params, f)
print("Saved Parameters!")


