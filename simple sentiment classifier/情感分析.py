# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 09:35:21 2020

@author: Administrator
"""

'''语料与代码来自https://spaces.ac.cn/archives/3414'''
'''少量内容被我简化'''

import pandas as pd #导入Pandas
import numpy as np #导入Numpy
import jieba #导入结巴分词
import time
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Flatten
from keras.preprocessing.text import Tokenizer


######################################
#读取数据
path_data_1='E:\\吴恩达深度学习系列\\NLP实验\\'

neg=pd.read_excel(path_data_1+'neg.xls',header=None,index=None)
pos=pd.read_excel(path_data_1+'pos.xls',header=None,index=None) #读取训练语料完毕
comment = pd.read_excel(path_data_1+'sum.xls') #读入评论内容

######################################
#整理数据
pos['mark']=1
neg['mark']=0 #给训练语料贴上标签
pn=pd.concat([pos,neg],ignore_index=True) #合并语料
neglen=len(neg)
poslen=len(pos) #计算语料数目

cw = lambda x: list(jieba.cut(x)) #定义分词函数，默认精准分词，cut生成迭代器可以for loop
pn['words'] = pn[0].apply(cw)

#comment = pd.read_csv('a.csv', encoding='utf-8')
comment = comment[comment['rateContent'].notnull()] #仅读取非空评论
comment['words'] = comment['rateContent'].apply(cw) #评论分词 

d2v_train = pd.concat([pn['words'], comment['words']], ignore_index = True) 

w = [] #将所有词语整合在一起
for i in d2v_train:
    w.extend(i)
    
#keras的转换功能
num_words=5000
tokenizer = Tokenizer(num_words)#出现频率最高的5000个词
# This builds the word index
tokenizer.fit_on_texts(w)
#one_hot版本,把句子映射成一堆1，1个句子由很多1和更多的0组成，列名称是单词
#one_hot_results = tokenizer.texts_to_matrix(pn['words'], mode='binary')
# This turns strings into lists of integer indices.
sequences = tokenizer.texts_to_sequences(pn['words'])
#字典
word_index = tokenizer.word_index
pn['sent'] = sequences #拼接DF

#多了砍掉，最多50个词
maxlen = 50

print("Pad sequences (samples x time)")
pn['sent'] = list(sequence.pad_sequences(pn['sent'], maxlen=maxlen))

x = np.array(list(pn['sent']))[::2] #训练集,[::2]表示[起始索引:结束索引(不含):步长]
y = np.array(list(pn['mark']))[::2]
xt = np.array(list(pn['sent']))[1::2] #测试集
yt = np.array(list(pn['mark']))[1::2]
xa = np.array(list(pn['sent'])) #全集
ya = np.array(list(pn['mark']))

print('训练集样本量',x.shape)
#(10553, 50)
print('测试集样本量',xt.shape)
#(10552, 50)
#训练集标记分布
from collections import  Counter
print(Counter(y))


#embedding层映射低维度的列数
EMBEDDING_LEN=256
######################################

#训练模型
print('Build model...')
model = Sequential()
#### 由于 没有预训练，设置+1
model.add(Embedding(len(word_index)+1, EMBEDDING_LEN)) #Embedding(词典最大容量+1, 映射后的维度自定义)
model.add(LSTM(128)) # try using a GRU instead, for fun
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

start_time=time.localtime(time.time())
model.fit(x, y, batch_size=16, epochs=10) #训练时间为若干个小时
end_time=time.localtime(time.time())
print('start_time:',start_time)
print('end_time:',end_time)


#保存模型
model.save(path_data_1+'情感分析.h5')

#读取模型
model=load_model(path_data_1+'情感分析.h5')

classes = model.predict_classes(xt)
acc = accuracy_score(classes, yt)
print('Test accuracy:', acc)
#87%
