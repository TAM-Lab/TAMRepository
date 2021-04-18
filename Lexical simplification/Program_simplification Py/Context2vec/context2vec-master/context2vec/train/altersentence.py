#!/usr/bin/python
#coding=utf-8
import pandas as pd
from pandas import DataFrame
data = pd.read_csv('E:\Simplify\_Datasets\_Wikipedia\LSTM method\Wikipedia_ds_pos.csv',header = None ,index_col = None)
# print (data)
row_num = data.shape[0]#获取dataframe行数，shape[1]获取列数
#print row_num
for i in range(0,row_num):
	# print i
	sentence = data.iloc[i,1].lower()
	target_word = data.iloc[i,2].split('.')[0]
	# pos = sentence.find(target_word)
	replace_word = '[' + target_word +']'
	sentence = sentence.replace(target_word, replace_word)
	#print sentence
	data.iloc[i,1] = sentence
data.to_csv("E:\Simplify\_Datasets\_Wikipedia\LSTM method\sentence_altered.csv")