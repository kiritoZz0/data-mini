#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import math
# hmmlearn可以在安装numpy以后，再使用pip install hmmlearn安装
from hmmlearn import hmm

states = ["Eat", "Sleep"]##隐藏状态
n_states = len(states)##隐藏状态长度

observations = ["cry", "nospirit", "findmom"]##可观察的状态
n_observations = len(observations)##可观察序列的长度

start_probability = np.array([0.3, 0.7])##开始转移概率，即开始是Eat和Sleep的概率
##隐藏混淆矩阵，即Eat和Sleep之间的转换关系
transition_probability = np.array([
  [0.1, 0.9],
  [0.8, 0.2]
])

emission_probability = np.array([
  [0.7, 0.1, 0.2],
  [0.3, 0.5, 0.2]
])

model = hmm.MultinomialHMM(n_components=n_states)
model.startprob_= start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability

#给出一个可见序列
bob_Actions = np.array([[0,1,2]]).T

# 解决问题1,解码问题,已知模型参数和X，估计最可能的Z； 维特比算法
logprob, nextaction = model.decode(bob_Actions, algorithm="viterbi")
list1=map(lambda x: observations[x], bob_Actions)
#print(list1)
# 解决问题2,概率问题，已知模型参数和X，估计X出现的概率, 向前-向后算法
score = (model.score(bob_Actions, lengths=None))
print("when baby's actions:cry ,",  "nospirit, findmom")
print('probability:',math.e**score)
print("baby's state:", ", ".join(map(lambda x: states[x], nextaction)))



